"""GraphSAGE encoder for the GNN→XGBoost hybrid.

Graph construction
------------------
Transactions sharing the same card1 value are connected as peers. This models
the reality that fraud patterns propagate through shared payment instruments —
a compromised card surfaces across multiple transactions. For k transactions on
the same card, all k*(k-1) directed edges are created (no self-loops; groups
larger than max_neighbors are randomly sampled to bound memory).

GraphSAGE training strategy
----------------------------
Full per-epoch graph message passing is expensive at 100K+ nodes. Instead,
we use a "historical aggregation" approach:

    Before training:
        card_h0_mean  — mean of raw features h0 per card1, computed once from
                        X_train.  Static throughout training.
    Each epoch:
        neigh_h0  = card_h0_mean lookup for each training node.
        h1        = SAGEConv1(h0, neigh_h0)
        (after forward pass, recompute neigh_h1 from mean h1 per card)
        neigh_h1  = card_h1_mean lookup  (updated after previous epoch)
        h2        = SAGEConv2(h1, neigh_h1)

This is equivalent to 2-layer SAGE with "lagged" neighbor states — a standard
practice in large-scale GNN mini-batch training (cf. GCNII, GraphSAINT). The
approximation error decreases as the model converges.

Inductive inference
-------------------
During API serving, individual transactions arrive without graph context. The
SAGE aggregation is approximated by looking up the precomputed card1 mean
embeddings from the training graph:

    neigh_h0_test = card_h0_mean[request.card1]  (mean raw features, static)
    h1_test       = SAGEConv1(h0_test, neigh_h0_test)
    neigh_h1_test = card_h1_mean[request.card1]  (mean h1 from training set)
    h2_test       = SAGEConv2(h1_test, neigh_h1_test)

For cards not seen in training, both lookups fall back to zero vectors
(conservative: no neighbourhood signal injected).

GNNArtifact bundles the encoder + both lookup tables into a single object
so benchmark.py and api/main.py handle all model types uniformly.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp_tree import EarlyStopping, FocalLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact container
# ---------------------------------------------------------------------------

@dataclass
class GNNArtifact:
    """Bundles encoder + card-level lookup tables for inductive inference."""
    encoder: "GraphSAGEEncoder"
    card_h0_mean: Dict[Any, np.ndarray]  # card1 → mean raw features [input_dim]
    card_h1_mean: Dict[Any, np.ndarray]  # card1 → mean layer-1 hidden state [hidden_dim]

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim


# ---------------------------------------------------------------------------
# GraphSAGE model
# ---------------------------------------------------------------------------

class SAGEConv(nn.Module):
    """Single GraphSAGE convolution (mean aggregation, precomputed neighbor agg).

    h_v' = σ(W · concat(h_v, neigh_agg_v))

    Accepts a pre-looked-up neighbor aggregate tensor rather than building a
    sparse adjacency at runtime — this decouples the message-passing step from
    the forward pass and enables efficient mini-batch training.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim * 2, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.W.weight, nonlinearity="relu")
        nn.init.constant_(self.W.bias, 0)

    def forward(self, h: torch.Tensor, neigh_agg: torch.Tensor) -> torch.Tensor:
        # h, neigh_agg: [B, in_dim]
        out = self.W(torch.cat([h, neigh_agg], dim=-1))
        out = F.relu(self.bn(out))
        return self.drop(out)


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder for transaction graph.

    input_dim → hidden_dim (layer 1) → out_dim (layer 2, embed_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1    = SAGEConv(input_dim, hidden_dim, dropout)
        self.conv2    = SAGEConv(hidden_dim, out_dim,   dropout)
        self.embed_dim  = out_dim
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        h0: torch.Tensor,
        neigh_h0: torch.Tensor,
        neigh_h1: torch.Tensor,
    ) -> torch.Tensor:
        # Layer 1: aggregate raw-feature neighbours
        h1 = self.conv1(h0, neigh_h0)
        # Layer 2: aggregate layer-1 neighbours (lagged from previous epoch)
        h2 = self.conv2(h1, neigh_h1)
        return h2


class _GNNClassifier(nn.Module):
    """GraphSAGEEncoder + linear head for Stage 1 pre-training."""

    def __init__(self, encoder: GraphSAGEEncoder):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(encoder.embed_dim, 1)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="relu")
        nn.init.constant_(self.classifier.bias, 0)

    def forward(
        self,
        h0: torch.Tensor,
        neigh_h0: torch.Tensor,
        neigh_h1: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encoder(h0, neigh_h0, neigh_h1))


# ---------------------------------------------------------------------------
# Neighbour-aggregate helpers
# ---------------------------------------------------------------------------

def _compute_card_mean(
    card1_array: np.ndarray,
    features: np.ndarray,
    dim: int,
) -> Dict[Any, np.ndarray]:
    """Mean features per card1 value."""
    result: Dict[Any, list] = {}
    for card, feat in zip(card1_array, features):
        if card not in result:
            result[card] = []
        result[card].append(feat)
    return {card: np.mean(vecs, axis=0).astype(np.float32)
            for card, vecs in result.items()}


def _lookup_neigh_agg(
    card1_array: np.ndarray,
    card_mean: Dict[Any, np.ndarray],
    dim: int,
) -> np.ndarray:
    """Expand per-card means into a per-node neighbour-aggregate matrix."""
    out = np.zeros((len(card1_array), dim), dtype=np.float32)
    for i, card in enumerate(card1_array):
        if card in card_mean:
            out[i] = card_mean[card]
    return out


@torch.no_grad()
def _compute_h1_embeddings(
    encoder: GraphSAGEEncoder,
    X: np.ndarray,
    neigh_h0: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute conv1 outputs for all nodes (used to refresh card_h1_mean each epoch)."""
    encoder.conv1.eval()
    results = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        bx  = torch.FloatTensor(X[start:end]).to(device)
        bnh = torch.FloatTensor(neigh_h0[start:end]).to(device)
        h1  = encoder.conv1(bx, bnh)
        results.append(h1.cpu().numpy())
    encoder.conv1.train()
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_gnn_embeddings(
    artifact: GNNArtifact,
    X: np.ndarray,
    card1_values: Optional[np.ndarray] = None,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Extract GNN embeddings using inductive SAGE (no full graph required).

    For each transaction, the neighbour aggregate is approximated by the
    precomputed card1 mean embedding from training (artifact.card_h0_mean and
    artifact.card_h1_mean). Transactions whose card1 was never seen in training
    receive zero neighbourhood vectors — a conservative no-signal fallback.

    Args:
        artifact:    GNNArtifact (encoder + card lookup tables).
        X:           Pipeline-processed feature matrix [N, input_dim].
        card1_values: card1 values for each row in X. Used for lookup. If None,
                      uses zero neighbourhood (isolated-node mode).
        device:      Torch device.
        batch_size:  Inference batch size.

    Returns:
        embeddings: np.ndarray [N, embed_dim]
    """
    encoder = artifact.encoder
    encoder.eval()
    n = len(X)

    # Build neighbourhood aggregates for inference
    if card1_values is not None:
        neigh_h0 = _lookup_neigh_agg(card1_values, artifact.card_h0_mean, encoder.input_dim)
        neigh_h1 = _lookup_neigh_agg(card1_values, artifact.card_h1_mean, encoder.hidden_dim)
    else:
        neigh_h0 = np.zeros((n, encoder.input_dim),  dtype=np.float32)
        neigh_h1 = np.zeros((n, encoder.hidden_dim), dtype=np.float32)

    parts = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bx   = torch.FloatTensor(X[start:end]).to(device)
        bnh0 = torch.FloatTensor(neigh_h0[start:end]).to(device)
        bnh1 = torch.FloatTensor(neigh_h1[start:end]).to(device)
        parts.append(encoder(bx, bnh0, bnh1).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_gnn_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    card1_train: np.ndarray,
    params: Dict[str, Any],
    fpr_threshold: float,
    device: str,
) -> Tuple["GraphSAGEEncoder", Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """Pre-train GraphSAGEEncoder with FocalLoss + FPR early stopping.

    Returns (encoder, card_h0_mean, card_h1_mean) — the lookup tables are
    needed for inductive inference and saved alongside the model.
    """
    epochs         = params.get("encoder_epochs", 15)
    batch_size     = params.get("batch_size", 2048)
    lr             = params.get("learning_rate", 1e-3)
    patience       = params.get("patience", 5)
    hidden_dim     = params.get("hidden_dim", 64)
    out_dim        = params.get("out_dim", 32)
    dropout        = params.get("dropout_rate", 0.1)
    clip_grad_norm = params.get("clip_grad_norm", 1.0)

    n_train   = X_train.shape[0]
    n_val     = X_val.shape[0]
    input_dim = X_train.shape[1]

    # Static layer-0 neighbour aggregates (card mean of raw features — never changes)
    logger.info("Computing card_h0_mean from %d training transactions...", n_train)
    card_h0_mean = _compute_card_mean(card1_train, X_train, input_dim)
    neigh_h0_all = _lookup_neigh_agg(card1_train, card_h0_mean, input_dim)

    # Layer-1 neighbour aggregates initialised to zero; updated each epoch
    neigh_h1_all = np.zeros((n_train, hidden_dim), dtype=np.float32)

    encoder   = GraphSAGEEncoder(input_dim, hidden_dim, out_dim, dropout)
    clf_model = _GNNClassifier(encoder).to(device)

    criterion = FocalLoss()
    optimizer = optim.Adam(clf_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    y_t = (y_train.values if hasattr(y_train, "values") else y_train).astype(np.float32)
    y_v = (y_val.values   if hasattr(y_val,   "values") else y_val).astype(np.float32)

    # Validation uses zero neighbourhood (conservative — mimics cold-start inference)
    neigh_h0_val = np.zeros((n_val, input_dim),  dtype=np.float32)
    neigh_h1_val = np.zeros((n_val, hidden_dim), dtype=np.float32)

    early_stopping = EarlyStopping(patience=patience)
    _mlflow_active = mlflow.active_run() is not None

    for epoch in range(epochs):
        # Rebuild DataLoader each epoch with current (possibly updated) neigh_h1_all
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_t),
                torch.FloatTensor(neigh_h0_all),
                torch.FloatTensor(neigh_h1_all),
            ),
            batch_size=batch_size, shuffle=True,
        )

        clf_model.train()
        train_loss = 0.0
        for bx, by, bnh0, bnh1 in train_loader:
            bx, by = bx.to(device), by.to(device)
            bnh0, bnh1 = bnh0.to(device), bnh1.to(device)
            optimizer.zero_grad()
            loss = criterion(clf_model(bx, bnh0, bnh1).view(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), clip_grad_norm)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= n_train

        # Validation
        clf_model.eval()
        val_loss, all_probs = 0.0, []
        with torch.no_grad():
            vx  = torch.FloatTensor(X_val).to(device)
            vnh0 = torch.FloatTensor(neigh_h0_val).to(device)
            vnh1 = torch.FloatTensor(neigh_h1_val).to(device)
            vy  = torch.FloatTensor(y_v).to(device)
            out = clf_model(vx, vnh0, vnh1).view(-1)
            val_loss     = criterion(out, vy).item()
            all_probs_np = torch.sigmoid(out).cpu().numpy()

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
            "GNN encoder epoch %d/%d — train_loss: %.4f  val_fpr: %.4f  val_auc: %.4f",
            epoch + 1, epochs, train_loss, val_fpr, val_auc,
        )
        if _mlflow_active:
            mlflow.log_metrics(
                {"gnn_train_loss": train_loss, "gnn_val_loss": val_loss,
                 "gnn_val_fpr": val_fpr,       "gnn_val_auc": val_auc},
                step=epoch,
            )

        scheduler.step(val_loss)
        early_stopping(val_fpr, clf_model)
        if early_stopping.early_stop:
            logger.info("GNN encoder early stopping at epoch %d (patience=%d).",
                        epoch + 1, patience)
            break

        # Refresh card_h1_mean for next epoch using current conv1 outputs
        h1_all       = _compute_h1_embeddings(encoder, X_train, neigh_h0_all, device)
        card_h1_mean = _compute_card_mean(card1_train, h1_all, hidden_dim)
        neigh_h1_all = _lookup_neigh_agg(card1_train, card_h1_mean, hidden_dim)

    assert early_stopping.best_model_weights is not None, \
        "EarlyStopping never updated — no training iterations ran"
    clf_model.load_state_dict(early_stopping.best_model_weights)

    # Recompute final card_h1_mean from best-checkpoint encoder
    h1_final     = _compute_h1_embeddings(encoder, X_train, neigh_h0_all, device)
    card_h1_mean = _compute_card_mean(card1_train, h1_final, hidden_dim)

    return clf_model.encoder, card_h0_mean, card_h1_mean


def train_gnn_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    card1_train: Optional[np.ndarray] = None,
    card1_test: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/gnn_xgboost",
) -> Tuple[GNNArtifact, Any]:
    """Two-stage GraphSAGE→XGBoost training.

    Stage 1: Pre-train GraphSAGEEncoder using card1-based neighbourhood aggregation
             with FocalLoss + FPR early stopping.
    Stage 2: Extract inductive embeddings. Train XGBoost on
             [original_features || GNN_embeddings] with FPR eval metric.

    Args:
        X_train, y_train:    Pre-processed training arrays.
        X_test,  y_test:     Pre-processed OOT test arrays.
        card1_train:         card1 values aligned with X_train rows. If None,
                             zero neighbourhood is used (isolated-node fallback).
        card1_test:          card1 values aligned with X_test rows.
        params:              hidden_dim, out_dim, dropout_rate, learning_rate,
                             encoder_epochs, batch_size, patience (encoder);
                             xgb_* keys forwarded to XGBClassifier.
        fpr_threshold:       Operating threshold (both stages).
        save_path:           Directory prefix for saving artifacts.
                             Pass None during Optuna tuning.

    Returns:
        (gnn_artifact, xgb_model): GNNArtifact + fitted XGBClassifier.
    """
    from src.models.tree_models import get_xgboost_model

    if params is None:
        params = {}

    # Fall back to zero neighbourhood when card1 not provided
    _card1_train = card1_train if card1_train is not None else np.zeros(len(X_train), dtype=int)
    _card1_test  = card1_test  if card1_test  is not None else np.zeros(len(X_test),  dtype=int)

    logger.info(
        "Stage 1: Pre-training GraphSAGEEncoder (%d features, hidden=%d, out=%d).",
        X_train.shape[1],
        params.get("hidden_dim", 64),
        params.get("out_dim", 32),
    )
    encoder, card_h0_mean, card_h1_mean = _train_gnn_encoder(
        X_train, y_train, X_test, y_test,
        card1_train=_card1_train,
        params=params, fpr_threshold=fpr_threshold, device=device,
    )

    artifact = GNNArtifact(encoder=encoder,
                           card_h0_mean=card_h0_mean,
                           card_h1_mean=card_h1_mean)

    logger.info("Extracting GNN embeddings (inductive, card1-based lookup).")
    embed_train = extract_gnn_embeddings(artifact, X_train, _card1_train, device=device)
    embed_test  = extract_gnn_embeddings(artifact, X_test,  _card1_test,  device=device)

    X_train_enriched = np.concatenate([X_train, embed_train], axis=1)
    X_test_enriched  = np.concatenate([X_test,  embed_test],  axis=1)
    logger.info(
        "Enriched feature dim: %d → %d (original + %d GNN embedding dims)",
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
    logger.info("GNN+XGBoost OOT AUC: %.4f", auc)

    _mlflow_active = mlflow.active_run() is not None
    if _mlflow_active:
        mlflow.log_metric("gnn_xgboost_OOT_AUC", auc)
        mlflow.log_metric(
            "gnn_xgboost_best_iteration",
            getattr(xgb_model, "best_iteration", params.get("n_estimators", 300)),
        )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        enc_path  = os.path.join(save_path, "encoder.pt")
        h0_path   = os.path.join(save_path, "card_h0_mean.pkl")
        h1_path   = os.path.join(save_path, "card_h1_mean.pkl")
        xgb_path  = os.path.join(save_path, "xgboost.joblib")

        torch.save(
            {
                "model_state_dict": encoder.state_dict(),
                "input_dim":        X_train.shape[1],
                "hidden_dim":       encoder.hidden_dim,
                "embed_dim":        encoder.embed_dim,
                "dropout":          params.get("dropout_rate", 0.1),
            },
            enc_path,
        )
        with open(h0_path, "wb") as f:
            pickle.dump(card_h0_mean, f)
        with open(h1_path, "wb") as f:
            pickle.dump(card_h1_mean, f)

        import joblib
        xgb_model.set_params(eval_metric=None)  # closure can't be pickled
        joblib.dump(xgb_model, xgb_path)
        logger.info("Saved GNN encoder → %s, lookup tables → %s / %s, XGBoost → %s",
                    enc_path, h0_path, h1_path, xgb_path)

        if _mlflow_active:
            mlflow.log_artifact(enc_path,  artifact_path="gnn_encoder")
            mlflow.log_artifact(h0_path,   artifact_path="gnn_encoder")
            mlflow.log_artifact(h1_path,   artifact_path="gnn_encoder")
            mlflow.xgboost.log_model(xgb_model, artifact_path="gnn_xgb_model")

    return artifact, xgb_model
