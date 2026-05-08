"""TabTransformer-style encoder for the Transformer→XGBoost hybrid.

Architecture (FT-Transformer variant)
--------------------------------------
Each scalar feature value is independently projected to d_model via a shared
linear layer, producing a sequence of "feature tokens". These tokens are
processed by TransformerEncoderLayers (multi-head self-attention). The final
embedding is the mean of all token outputs — dimension d_model.

This approach captures high-order feature interactions via attention without
requiring an explicit categorical/continuous split, making it compatible with
the existing sklearn pipeline output (all-numeric array).

Pre-LayerNorm (norm_first=True) and CosineAnnealingLR are used for training
stability; both are well-established practices for transformer fine-tuning on
tabular data.

Two-stage hybrid: same interface as mlp_tree.train_mlp_xgboost.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.training.models.mlp_tree import EarlyStopping, FocalLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TabTransformerEncoder(nn.Module):
    """FT-Transformer style encoder: per-feature projection + attention + mean pool.

    Each feature value is projected to d_model (treating the input as a sequence
    of feature "tokens"). TransformerEncoderLayers apply self-attention across
    feature tokens. Mean pooling over the feature dimension produces a
    fixed-size embedding of dimension d_model.

    The positional encoding is learnable (per-feature, not per-position) so the
    model can assign different importance to different feature slots even after
    the linear projection equalises their scale.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project each scalar feature to d_model (shared weight across feature slots)
        self.feature_proj = nn.Linear(1, d_model)

        # Learnable per-feature positional encoding — lets the model distinguish slots
        self.pos_enc = nn.Parameter(torch.zeros(input_dim, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable gradient flow than Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.embed_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        tokens = self.feature_proj(x.unsqueeze(-1))  # [B, input_dim, d_model]
        tokens = tokens + self.pos_enc               # add per-feature positional encoding
        out = self.transformer(tokens)               # [B, input_dim, d_model]
        return out.mean(dim=1)                       # [B, d_model] — mean pool over feature tokens


class _TransformerClassifier(nn.Module):
    """TabTransformerEncoder + linear head for Stage 1 pre-training."""

    def __init__(self, encoder: TabTransformerEncoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embed_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_transformer_embeddings(
    encoder: TabTransformerEncoder,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Extract embeddings from a fitted TabTransformerEncoder.

    Same interface as extract_mlp_embeddings from mlp_tree.

    Returns:
        embeddings: np.ndarray [N, d_model]
    """
    encoder.eval()
    parts = []
    for start in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[start: start + batch_size]).to(device)
        parts.append(encoder(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_transformer_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    fpr_threshold: float,
    device: str,
) -> TabTransformerEncoder:
    """Pre-train TabTransformerEncoder with FocalLoss + FPR early stopping.

    Returns the encoder (no classification head) with best-checkpoint weights.
    """
    epochs         = params.get("encoder_epochs", 20)
    batch_size     = params.get("batch_size", 512)
    lr             = params.get("learning_rate", 5e-4)
    patience       = params.get("patience", 5)
    d_model        = params.get("d_model", 64)
    nhead          = params.get("nhead", 4)
    num_layers     = params.get("num_layers", 2)
    dim_ff         = params.get("dim_feedforward", 256)
    dropout        = params.get("dropout_rate", 0.1)
    clip_grad_norm = params.get("clip_grad_norm", 1.0)

    input_dim = X_train.shape[1]
    encoder   = TabTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
    )
    clf_model = _TransformerClassifier(encoder).to(device)

    criterion = FocalLoss()
    optimizer = optim.AdamW(clf_model.parameters(), lr=lr, weight_decay=1e-4)
    # CosineAnnealingLR matches the transformer training schedule better than ReduceLROnPlateau
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    y_t = y_train.values if hasattr(y_train, "values") else y_train
    y_v = y_val.values   if hasattr(y_val,   "values") else y_val
    n_train = X_train.shape[0]
    n_val   = X_val.shape[0]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_t)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_v)),
        batch_size=batch_size * 2, shuffle=False,
    )

    early_stopping = EarlyStopping(patience=patience)
    _mlflow_active = mlflow.active_run() is not None

    for epoch in range(epochs):
        clf_model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(clf_model(bx).view(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), clip_grad_norm)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= n_train
        scheduler.step()

        clf_model.eval()
        val_loss, all_probs = 0.0, []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = clf_model(bx).view(-1)
                val_loss  += criterion(out, by).item() * bx.size(0)
                all_probs.append(torch.sigmoid(out).cpu().numpy())
        val_loss    /= n_val
        all_probs_np = np.concatenate(all_probs)

        preds    = (all_probs_np >= fpr_threshold).astype(int)
        neg_mask = y_v == 0
        fp       = int(((preds == 1) & neg_mask).sum())
        tn       = int(((preds == 0) & neg_mask).sum())
        val_fpr  = fp / (fp + tn + 1e-8)
        try:
            val_auc = roc_auc_score(y_v, all_probs_np)
        except ValueError:
            val_auc = 0.5

        logger.info(
            "Transformer encoder epoch %d/%d — train_loss: %.4f  val_fpr: %.4f  val_auc: %.4f",
            epoch + 1, epochs, train_loss, val_fpr, val_auc,
        )
        if _mlflow_active:
            mlflow.log_metrics(
                {"tab_enc_train_loss": train_loss, "tab_enc_val_loss": val_loss,
                 "tab_enc_val_fpr": val_fpr, "tab_enc_val_auc": val_auc},
                step=epoch,
            )

        early_stopping(val_fpr, clf_model)
        if early_stopping.early_stop:
            logger.info("Transformer encoder early stopping at epoch %d (patience=%d).",
                        epoch + 1, patience)
            break

    assert early_stopping.best_model_weights is not None, \
        "EarlyStopping never updated — no training iterations ran"
    clf_model.load_state_dict(early_stopping.best_model_weights)
    return clf_model.encoder


def train_transformer_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/transformer_xgboost",
) -> Tuple[TabTransformerEncoder, Any]:
    """Two-stage Transformer→XGBoost training.

    Stage 1: Pre-train TabTransformerEncoder with FocalLoss + FPR early stopping.
    Stage 2: Extract embeddings. Train XGBoost on
             [original_features || Transformer_embeddings] with FPR eval metric.

    Same interface as train_mlp_xgboost. Drop-in substitutable in tune.py.

    Args:
        X_train, y_train: Pre-processed training arrays.
        X_test,  y_test:  Pre-processed OOT test arrays.
        params:   d_model, nhead, num_layers, dim_feedforward, dropout_rate,
                  learning_rate, encoder_epochs, batch_size, patience (encoder);
                  xgb_* keys forwarded to XGBClassifier.
        fpr_threshold: Operating threshold (both stages).
        save_path:     Directory prefix for saving encoder + XGBoost artifacts.
                       Pass None during Optuna tuning.

    Returns:
        (encoder, xgb_model): TabTransformerEncoder + fitted XGBClassifier.
    """
    from src.training.models.tree_models import get_xgboost_model

    if params is None:
        params = {}

    logger.info(
        "Stage 1: Pre-training TabTransformerEncoder (%d features, d_model=%d, nhead=%d, layers=%d).",
        X_train.shape[1],
        params.get("d_model", 64),
        params.get("nhead", 4),
        params.get("num_layers", 2),
    )
    encoder = _train_transformer_encoder(
        X_train, y_train, X_test, y_test,
        params=params, fpr_threshold=fpr_threshold, device=device,
    )

    logger.info("Extracting Transformer embeddings from best-checkpoint encoder.")
    embed_train = extract_transformer_embeddings(encoder, X_train, device=device)
    embed_test  = extract_transformer_embeddings(encoder, X_test,  device=device)

    X_train_enriched = np.concatenate([X_train, embed_train], axis=1)
    X_test_enriched  = np.concatenate([X_test,  embed_test],  axis=1)
    logger.info(
        "Enriched feature dim: %d → %d (original + %d Transformer embedding dims)",
        X_train.shape[1], X_train_enriched.shape[1], embed_train.shape[1],
    )

    xgb_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "min_child_weight",
        "reg_alpha", "reg_lambda", "gamma", "tree_method",
    }
    xgb_params = {k: v for k, v in params.items() if k in xgb_keys}
    xgb_early_stopping = params.get("xgb_early_stopping_rounds", 30)

    logger.info("Stage 2: Training XGBoost on %d enriched features with FPR early stopping.",
                X_train_enriched.shape[1])
    xgb_model = get_xgboost_model(
        params=xgb_params or None,
        early_stopping_rounds=xgb_early_stopping,
        fpr_threshold=fpr_threshold,
    )
    xgb_model.fit(
        X_train_enriched, y_train,
        eval_set=[(X_test_enriched, y_test)],
        verbose=False,
    )

    from sklearn.metrics import roc_auc_score as _roc_auc
    preds = xgb_model.predict_proba(X_test_enriched)[:, 1]
    auc   = _roc_auc(y_test, preds)
    logger.info("Transformer+XGBoost OOT AUC: %.4f", auc)

    _mlflow_active = mlflow.active_run() is not None
    if _mlflow_active:
        mlflow.log_metric("transformer_xgboost_OOT_AUC", auc)
        mlflow.log_metric(
            "transformer_xgboost_best_iteration",
            getattr(xgb_model, "best_iteration", params.get("n_estimators", 300)),
        )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        enc_path = os.path.join(save_path, "encoder.pt")
        xgb_path = os.path.join(save_path, "xgboost.joblib")

        torch.save(
            {
                "model_state_dict": encoder.state_dict(),
                "input_dim":        X_train.shape[1],
                "d_model":          encoder.d_model,
                "embed_dim":        encoder.embed_dim,
                "nhead":            params.get("nhead", 4),
                "num_layers":       params.get("num_layers", 2),
                "dim_feedforward":  params.get("dim_feedforward", 256),
                "dropout":          params.get("dropout_rate", 0.1),
            },
            enc_path,
        )
        import joblib
        xgb_model.set_params(eval_metric=None)  # closure can't be pickled
        joblib.dump(xgb_model, xgb_path)
        logger.info("Saved Transformer encoder → %s, XGBoost → %s", enc_path, xgb_path)

        if _mlflow_active:
            mlflow.log_artifact(enc_path, artifact_path="transformer_encoder")
            mlflow.xgboost.log_model(xgb_model, artifact_path="transformer_xgb_model")

    return encoder, xgb_model
