"""Unified neural network module for fraud detection.

FPR-based early stopping
------------------------
The training loop monitors FPR at the operating threshold (not val_loss) to
align training directly with the business constraint (FP rate < 15%).
FPR at a high threshold plateaus faster than loss, so training stops earlier
while optimising the metric that actually matters in production.

The ReduceLROnPlateau scheduler still uses val_loss — this is intentional:
LR decay responds to the loss landscape (smooth gradient signal), while
early stopping responds to the business metric (discrete, fast-plateauing).

MLP→XGBoost hybrid (train_mlp_xgboost)
---------------------------------------
Two-stage training:
  Stage 1 — MLPEncoder pre-trained with FocalLoss + FPR early stopping.
  Stage 2 — XGBoost trained on [original_features || MLP_embeddings].
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced binary classification.

    Standard BCE is minimised by predicting "not fraud" for everything at
    96%/4% imbalance.  Focal Loss down-weights easy (high-confidence)
    examples so the model's gradient signal is dominated by the hard,
    uncertain cases — disproportionately fraud.

        FL(pt) = alpha * (1 - pt)^gamma * BCE(pt)

    alpha=0.25: reduces weight on easy negatives
    gamma=2.0:  at pt=0.95 → modulating factor ≈ 0.0025 (near-zero loss)
                at pt=0.50 → modulating factor = 0.25  (full loss)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  raw logits (before sigmoid)
            targets: binary labels [0, 1]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving (lower = better).

    Saves a deepcopy of the best model weights so the caller can restore
    the best checkpoint after the loop ends — not the last (potentially
    overfit) checkpoint.

    Works for both val_loss and FPR since both are minimised metrics.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric: Optional[float] = None
        self.early_stop = False
        self.best_model_weights: Optional[Dict[str, Any]] = None

    def __call__(self, metric: float, model: nn.Module) -> None:
        if self.best_metric is None:
            self.best_metric = metric
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif metric > self.best_metric - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = metric
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0


# ---------------------------------------------------------------------------
# MLP→XGBoost hybrid
# ---------------------------------------------------------------------------

class _ResidualLayer(nn.Module):
    """Single MLP layer with an optional residual (skip) connection.

    When in_dim != out_dim a linear projection aligns dimensions for the
    residual path (same convention as ResNet bottleneck blocks).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Identity shortcut when dims match; linear projection otherwise.
        self.shortcut: nn.Module = (
            nn.Identity() if in_dim == out_dim
            else nn.Linear(in_dim, out_dim, bias=False)
        )
        nn.init.kaiming_normal_(self.main[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.main[0].bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_normal_(self.shortcut.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class MLPEncoder(nn.Module):
    """MLP feature extractor for the MLP→XGBoost hybrid.

    Architecture: input → [256 → 128 → 64] (default)
    Each layer: Linear → BatchNorm → ReLU → Dropout
    Optional residual connections via use_residual=True.

    ReLU is used because the embedding output feeds XGBoost, not another
    neural layer — dead neurons matter less when we are not backpropagating
    through the embedding at Stage 2.

    Hidden dims are configurable so Optuna can compare [256,128,64] vs
    [512,256,128] without changing any code. The last element of hidden_dims
    is the embedding dimension passed to XGBoost.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.3,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        layers: List[nn.Module] = []
        in_d = input_dim
        for out_d in hidden_dims:
            if use_residual:
                layers.append(_ResidualLayer(in_d, out_d, dropout_rate))
            else:
                layers += [
                    nn.Linear(in_d, out_d),
                    nn.BatchNorm1d(out_d),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            in_d = out_d
        self.encoder  = nn.Sequential(*layers)
        self.embed_dim = hidden_dims[-1]
        if not use_residual:
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # [B, embed_dim]


class _MLPEncoderClassifier(nn.Module):
    """MLPEncoder + linear head for Stage 1 pre-training."""

    def __init__(self, encoder: MLPEncoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(encoder.embed_dim, 1)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


@torch.no_grad()
def extract_mlp_embeddings(
    encoder: MLPEncoder,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Extract embeddings from a fitted MLPEncoder.

    Runs in no_grad mode — inference only. Processes in batches to handle
    arrays larger than GPU memory.

    Returns:
        embeddings: np.ndarray [N, embed_dim]
    """
    encoder.eval()
    parts = []
    for start in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[start: start + batch_size]).to(device)
        parts.append(encoder(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _train_mlp_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    fpr_threshold: float,
    device: str,
) -> MLPEncoder:
    """Pre-train MLPEncoder with FocalLoss, ReduceLROnPlateau, gradient clipping,
    and FPR-based early stopping.

    Returns the encoder (no classification head) with best-checkpoint weights.
    """
    epochs          = params.get("encoder_epochs", 20)
    batch_size      = params.get("batch_size", 1024)
    lr              = params.get("learning_rate", 1e-3)
    patience        = params.get("patience", 5)
    hidden_dims     = tuple(params.get("hidden_dims", [256, 128, 64]))
    use_residual    = params.get("use_residual", False)
    clip_grad_norm  = params.get("clip_grad_norm", 1.0)

    input_dim = X_train.shape[1]
    encoder   = MLPEncoder(
        input_dim, hidden_dims=hidden_dims,
        dropout_rate=params.get("dropout_rate", 0.3),
        use_residual=use_residual,
    )
    clf_model = _MLPEncoderClassifier(encoder).to(device)

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
            # Gradient clipping prevents exploding gradients in deep residual networks.
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), max_norm=clip_grad_norm)
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
            "MLP encoder epoch %d/%d — train_loss: %.4f  val_fpr: %.4f  val_auc: %.4f",
            epoch + 1, epochs, train_loss, val_fpr, val_auc,
        )
        if _mlflow_active:
            mlflow.log_metrics(
                {"enc_train_loss": train_loss, "enc_val_loss": val_loss,
                 "enc_val_fpr": val_fpr, "enc_val_auc": val_auc},
                step=epoch,
            )

        # LR scheduler uses val_loss (smooth signal); early stopping uses FPR (business metric).
        scheduler.step(val_loss)
        early_stopping(val_fpr, clf_model)
        if early_stopping.early_stop:
            logger.info(
                "MLP encoder early stopping at epoch %d (patience=%d).",
                epoch + 1, patience,
            )
            break

    assert early_stopping.best_model_weights is not None, "EarlyStopping never updated — no training iterations ran"
    clf_model.load_state_dict(early_stopping.best_model_weights)
    return clf_model.encoder


def train_mlp_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/mlp_xgboost",
) -> tuple:
    """Two-stage MLP→XGBoost training.

    Stage 1: Pre-train MLPEncoder with FocalLoss + FPR early stopping.
    Stage 2: Extract embeddings. Train XGBoost on
             [original_features || MLP_embeddings] with FPR eval metric.

    Args:
        X_train, y_train: Pre-processed training arrays.
        X_test,  y_test:  Pre-processed OOT test arrays.
        params:   hidden_dims, dropout_rate, learning_rate, encoder_epochs,
                  batch_size, patience, use_residual, clip_grad_norm (encoder);
                  xgb_* keys forwarded to XGBClassifier,
                  xgb_early_stopping_rounds.
        fpr_threshold: Operating threshold (both stages).
        save_path:     Directory prefix for saving encoder + XGBoost artifacts.
                       Pass None during Optuna tuning.

    Returns:
        (encoder, xgb_model): MLPEncoder + fitted XGBClassifier.
    """
    from src.training.models.tree_models import get_xgboost_model

    if params is None:
        params = {}

    # ---- Stage 1: encoder pre-training ----
    logger.info("Stage 1: Pre-training MLPEncoder (%d features, hidden_dims=%s, use_residual=%s).",
                X_train.shape[1], params.get("hidden_dims", [256, 128, 64]),
                params.get("use_residual", False))
    encoder = _train_mlp_encoder(
        X_train, y_train, X_test, y_test,
        params=params, fpr_threshold=fpr_threshold, device=device,
    )

    # ---- Embedding extraction ----
    logger.info("Extracting MLP embeddings from best-checkpoint encoder.")
    embed_train = extract_mlp_embeddings(encoder, X_train, device=device)
    embed_test  = extract_mlp_embeddings(encoder, X_test,  device=device)

    X_train_enriched = np.concatenate([X_train, embed_train], axis=1)
    X_test_enriched  = np.concatenate([X_test,  embed_test],  axis=1)
    logger.info(
        "Enriched feature dim: %d → %d (original + %d MLP embedding dims)",
        X_train.shape[1], X_train_enriched.shape[1], embed_train.shape[1],
    )

    # ---- Stage 2: XGBoost on enriched features ----
    xgb_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "min_child_weight",
        "reg_alpha", "reg_lambda", "gamma", "tree_method",
    }
    xgb_params = {k: v for k, v in params.items() if k in xgb_keys}
    xgb_early_stopping = params.get("xgb_early_stopping_rounds", 30)

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

    from sklearn.metrics import roc_auc_score as _roc_auc
    preds = xgb_model.predict_proba(X_test_enriched)[:, 1]
    auc   = _roc_auc(y_test, preds)
    logger.info("MLP+XGBoost OOT AUC: %.4f", auc)

    _mlflow_active = mlflow.active_run() is not None
    if _mlflow_active:
        mlflow.log_metric("mlp_xgboost_OOT_AUC", auc)
        mlflow.log_metric(
            "mlp_xgboost_best_iteration",
            getattr(xgb_model, "best_iteration", params.get("n_estimators", 300)),
        )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        enc_path = os.path.join(save_path, "encoder.pt")
        xgb_path = os.path.join(save_path, "xgboost.joblib")

        torch.save(
            {"model_state_dict": encoder.state_dict(),
             "input_dim": X_train.shape[1],
             "embed_dim": encoder.embed_dim},
            enc_path,
        )
        import joblib
        xgb_model.set_params(eval_metric=None)  # closure can't be pickled
        joblib.dump(xgb_model, xgb_path)
        logger.info("Saved MLP encoder → %s, XGBoost → %s", enc_path, xgb_path)

        if _mlflow_active:
            mlflow.log_artifact(enc_path, artifact_path="mlp_encoder")
            mlflow.log_artifact(xgb_path, artifact_path="mlp_xgboost")
            mlflow.xgboost.log_model(xgb_model, artifact_path="mlp_xgb_model")

    return encoder, xgb_model
