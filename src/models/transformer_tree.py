"""TabTransformer → XGBoost hybrid for fraud detection.

Architecture overview
---------------------
Stage 1 — TabTransformer encoder
    Each input feature is projected into a D-dimensional token by a learned
    linear map (FeatureTokenizer). A learnable CLS token is prepended and the
    sequence is passed through a standard TransformerEncoder. The CLS output
    aggregates global feature interactions into a fixed-length embedding.

Stage 2 — XGBoost classifier on enriched features
    The enriched feature vector [original_features || CLS_embedding] is fed
    to XGBoost. Concatenating originals preserves monotone raw signals (e.g.
    TransactionAmt) that the transformer may smooth out.

Why this hybrid?
----------------
- Transformer: captures high-order feature interactions that tree splits miss.
- XGBoost: production-proven, handles missing values natively, fast inference.
- SHAP: still works on the XGBoost layer — critical for regulatory explainability
  requirements in banking (Basel/SR 11-7 model risk governance).
- Two-stage training: encoder is pre-trained with a classification head using
  FocalLoss + FPR early stopping; XGBoost then trains on frozen embeddings.

Training
--------
- Pre-LN (norm_first=True) TransformerEncoder: more stable training than post-LN.
- FocalLoss for class imbalance during encoder pre-training.
- FPR-based early stopping on encoder pre-training (same pattern as mlp_tree).
- XGBoost second stage uses FPR eval metric from tree_models.make_fpr_eval_metric.
"""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.xgboost
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.models.mlp_tree import EarlyStopping, FocalLoss
from src.models.tree_models import get_xgboost_model, make_fpr_eval_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Project each scalar feature into a D-dimensional token.

    Each feature i gets its own weight vector w_i ∈ R^D and bias b_i ∈ R^D:
        token_i = x_i * w_i + b_i

    This is equivalent to an embedding table for continuous features.
    Kaiming init on weights (fan_in, linear nonlinearity).
    """

    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(num_features, d_token))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_features]
        Returns:
            tokens: [B, num_features, d_token]
        """
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabTransformer(nn.Module):
    """TabTransformer: tokenize features → prepend CLS → TransformerEncoder.

    The CLS token aggregates global feature interactions. We use only the CLS
    output as the enriched embedding — it is a fixed-length representation
    regardless of input dimensionality.

    Pre-LN (norm_first=True) is used for training stability: gradient norms
    are bounded at each layer, making learning rate selection less critical.
    """

    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        # Learnable CLS token — trunc_normal init for transformer tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=nhead,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable than post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_features]
        Returns:
            cls_emb: [B, d_token]  — CLS token output (global feature summary)
        """
        tokens = self.tokenizer(x)                            # [B, F, D]
        cls    = self.cls_token.expand(x.size(0), -1, -1)    # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)              # [B, 1+F, D]
        out    = self.encoder(tokens)                          # [B, 1+F, D]
        return self.norm(out[:, 0, :])                         # CLS: [B, D]


class _TabTransformerClassifier(nn.Module):
    """TabTransformer + linear classification head for pre-training."""

    def __init__(self, encoder: TabTransformer):
        super().__init__()
        self.encoder    = encoder
        d_token         = encoder.norm.normalized_shape[0]
        self.classifier = nn.Linear(d_token, 1)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in",
                                nonlinearity="leaky_relu")
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_transformer(params: Optional[Dict[str, Any]], input_dim: int) -> TabTransformer:
    """Instantiate a TabTransformer from a params dict.

    Mirrors get_xgboost_model / get_mlp_model for uniform factory pattern.
    """
    if params is None:
        params = {}
    return TabTransformer(
        num_features=input_dim,
        d_token=params.get("d_token", 64),
        nhead=params.get("nhead", 4),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout_rate", 0.1),
    )


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: TabTransformer,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Extract CLS embeddings from a fitted TabTransformer.

    Runs in no_grad mode — inference only, no gradient accumulation.
    Processes in batches to handle arrays larger than GPU memory.

    Returns:
        embeddings: np.ndarray [N, d_token]
    """
    encoder.eval()
    parts = []
    for start in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[start: start + batch_size]).to(device)
        parts.append(encoder(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Stage 1: pre-train TabTransformer encoder
# ---------------------------------------------------------------------------

def _train_transformer_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    fpr_threshold: float,
    device: str,
) -> TabTransformer:
    """Pre-train TabTransformer with FocalLoss and FPR-based early stopping.

    Returns the encoder (no classification head) with best-checkpoint weights.
    The head is discarded after pre-training; only the encoder is used in
    Stage 2 for embedding extraction.
    """
    epochs     = params.get("encoder_epochs", 30)
    batch_size = params.get("batch_size", 1024)
    lr         = params.get("learning_rate", 1e-3)
    patience   = params.get("patience", 5)

    input_dim = X_train.shape[1]
    encoder   = get_transformer(params, input_dim)
    clf_model = _TabTransformerClassifier(encoder).to(device)

    criterion = FocalLoss()
    optimizer = optim.Adam(clf_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    y_t = y_train.values if hasattr(y_train, "values") else y_train
    y_v = y_val.values   if hasattr(y_val,   "values") else y_val
    n_train = int(X_train.shape[0])
    n_val   = int(X_val.shape[0])

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_t)),
        batch_size=batch_size, shuffle=True,
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
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= n_train

        clf_model.eval()
        val_loss, all_probs = 0.0, []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = clf_model(bx).view(-1)
                val_loss  += criterion(out, by).item() * bx.size(0)
                all_probs.append(torch.sigmoid(out).cpu().numpy())
        val_loss     /= n_val
        all_probs_np  = np.concatenate(all_probs)

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
            "Transformer encoder epoch %d/%d — train_loss: %.4f  "
            "val_fpr: %.4f  val_auc: %.4f",
            epoch + 1, epochs, train_loss, val_fpr, val_auc,
        )
        if _mlflow_active:
            mlflow.log_metrics(
                {"enc_train_loss": train_loss, "enc_val_loss": val_loss,
                 "enc_val_fpr": val_fpr, "enc_val_auc": val_auc},
                step=epoch,
            )

        scheduler.step(val_loss)
        early_stopping(val_fpr, clf_model)
        if early_stopping.early_stop:
            logger.info(
                "Encoder early stopping at epoch %d (patience=%d).",
                epoch + 1, patience,
            )
            break

    # Restore best encoder weights (strip classification head)
    assert early_stopping.best_model_weights is not None, "EarlyStopping never updated — no training iterations ran"
    clf_model.load_state_dict(early_stopping.best_model_weights)
    return clf_model.encoder


# ---------------------------------------------------------------------------
# Stage 2: two-stage training (encoder → embeddings → XGBoost)
# ---------------------------------------------------------------------------

def train_transformer_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/transformer_xgboost",
) -> tuple:
    """Two-stage TabTransformer → XGBoost training.

    Stage 1: Pre-train TabTransformer encoder with FocalLoss + FPR early stopping.
    Stage 2: Extract CLS embeddings. Train XGBoost on
             [original_features || CLS_embeddings] with FPR eval metric.

    Concatenating original features ensures monotone raw signals (e.g.
    TransactionAmt) are not lost through the transformer's attention smoothing.

    Args:
        X_train, y_train: Pre-processed training arrays.
        X_test,  y_test:  Pre-processed OOT test arrays.
        params:   d_token, nhead, num_layers, dropout_rate, learning_rate,
                  encoder_epochs, batch_size, patience (encoder),
                  xgb_* keys forwarded to XGBClassifier,
                  xgb_early_stopping_rounds.
        fpr_threshold: Operating threshold from config (both stages).
        save_path:     Directory prefix for saving encoder + XGBoost artifacts.
                       Pass None during Optuna tuning.

    Returns:
        (encoder, xgb_model): TabTransformer encoder + fitted XGBClassifier.
        Both are needed for serving: encoder transforms inputs, XGBoost classifies.
    """
    if params is None:
        params = {}

    # ---- Stage 1: encoder pre-training ----
    logger.info("Stage 1: Pre-training TabTransformer encoder (%d features).",
                X_train.shape[1])
    encoder = _train_transformer_encoder(
        X_train, y_train, X_test, y_test,
        params=params,
        fpr_threshold=fpr_threshold,
        device=device,
    )

    # ---- Embedding extraction ----
    logger.info("Extracting CLS embeddings from best-checkpoint encoder.")
    embed_train = extract_embeddings(encoder, X_train, device=device)
    embed_test  = extract_embeddings(encoder, X_test,  device=device)

    # Enrich: [original || CLS embedding]
    X_train_enriched = np.concatenate([X_train, embed_train], axis=1)
    X_test_enriched  = np.concatenate([X_test,  embed_test],  axis=1)
    logger.info(
        "Enriched feature dim: %d → %d (original + %d CLS dims)",
        X_train.shape[1], X_train_enriched.shape[1], embed_train.shape[1],
    )

    # ---- Stage 2: XGBoost on enriched features ----
    xgb_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "min_child_weight",
        "reg_alpha", "reg_lambda", "gamma", "tree_method",
    }
    xgb_params = {k: v for k, v in params.items() if k in xgb_keys}
    xgb_early_stopping = params.get("xgb_early_stopping_rounds", 50)

    logger.info(
        "Stage 2: Training XGBoost on %d enriched features with FPR early stopping.",
        X_train_enriched.shape[1],
    )
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

    preds = xgb_model.predict_proba(X_test_enriched)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    logger.info("TabTransformer+XGBoost OOT AUC: %.4f", auc)

    _mlflow_active = mlflow.active_run() is not None
    if _mlflow_active:
        mlflow.log_metric("transformer_xgboost_OOT_AUC", auc)
        mlflow.log_metric(
            "transformer_xgboost_best_iteration",
            getattr(xgb_model, "best_iteration", params.get("n_estimators", 500)),
        )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        enc_path = os.path.join(save_path, "encoder.pt")
        xgb_path = os.path.join(save_path, "xgboost.joblib")

        torch.save(
            {"model_state_dict": encoder.state_dict(),
             "input_dim": X_train.shape[1]},
            enc_path,
        )
        import joblib
        joblib.dump(xgb_model, xgb_path)
        logger.info("Saved encoder → %s, XGBoost → %s", enc_path, xgb_path)

        if _mlflow_active:
            mlflow.log_artifact(enc_path,  artifact_path="transformer_encoder")
            mlflow.log_artifact(xgb_path,  artifact_path="transformer_xgboost")
            mlflow.xgboost.log_model(xgb_model, artifact_path="transformer_xgb_model")

    return encoder, xgb_model
