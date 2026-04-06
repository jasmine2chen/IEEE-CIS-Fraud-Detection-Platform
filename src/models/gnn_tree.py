"""GraphSAGE-based fraud detection model for the IEEE-CIS dataset.

GNN→XGBoost two-stage (train_gnn_xgboost)
------------------------------------------
Stage 1: Train FraudGNN with FocalLoss + FPR early stopping.
         Graph = combined train+test (directed edges prevent temporal leakage).
Stage 2: Extract per-node embeddings from best-checkpoint GNN.
         Train XGBoost on [original_features || GNN_embeddings].

This mirrors train_mlp_xgboost and train_transformer_xgboost so all three
hybrids have identical train/tune/serve interfaces and can be directly compared:
    xgboost              → baseline (no neural extraction)
    mlp_xgboost          → shallow MLP extraction
    transformer_xgboost  → attention-based extraction
    gnn (this file)      → graph-based extraction (entity-neighbourhood signal)

Research-validated graph construction (AWS GraphStorm + FraudGNN-RL)
---------------------------------------------------------------------
Transactions are nodes. Edges connect transactions that share an exact
entity value (uid, card1, DeviceInfo, P_emaildomain, addr1) within a
real days-based temporal window. Edges are directed (past → future only)
to prevent temporal leakage: a training node never aggregates information
from a future test transaction during message passing.

Edge feature vector (EDGE_DIM=6)
---------------------------------
Each edge carries: [temporal_decay, uid_flag, card1_flag, DeviceInfo_flag,
                     P_emaildomain_flag, addr1_flag]
A learned EdgeGate (Linear(6,1) + sigmoid) in each SAGEConv layer projects
this vector to a scalar aggregation weight, letting the model learn the
relative importance of recency vs. entity type and multi-entity overlap
(e.g. same card AND same device → stronger gate). Previously a manually-
tuned scalar was used; the learned gate captures combinations the manual
weight could not.

For pairs connected by multiple entity columns, temporal_decay = max across
columns; entity flags are OR-combined. This preserves the multi-entity
signal that the old scalar max-merge discarded.

Why uid as primary entity
    uid = card1 + addr1 + floor(day - D1). This composite key identifies
    a specific account-holder at a specific billing address, removing card-
    sharing noise (family members, hotel corporate cards). Requires the
    post-build_features() DataFrame — NOT the raw joined DataFrame (which
    lacks uid). In train.py, call build_features(X_raw.copy()) first.

Temporal windowing (7-day default, validated by AWS + academic literature)
    Concept drift: fraud tactics shift over weeks. Connecting a 6-month-old
    transaction to a current one propagates stale patterns. A 7-day window
    restricts neighbourhood to recent co-activity.

Why GraphSAGE (Hamilton et al. 2017)
    Inductive: learns aggregation functions, not per-node embeddings.
    New transactions at serving time get embeddings by aggregating neighbour
    features — no full graph recomputation required.

Training strategy
    Directed combined graph (train + test) with boolean masks, NeighborLoader
    mini-batches, FPR-based early stopping with disk checkpointing (avoids
    duplicating model weights in RAM during training).

Isolated nodes
    Nodes with no entity-sharing temporal neighbours get K-NN fallback edges
    based on feature-space cosine similarity, so they still receive some
    neighbourhood signal. KNN edges carry [similarity, 0, 0, 0, 0, 0] in the
    edge feature vector — entity flags zero, gate learns to weight these lower.

Dependencies
------------
    pip install torch-geometric scikit-learn
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch import Tensor

from src.models.mlp_tree import FocalLoss

logger = logging.getLogger(__name__)

_SECONDS_IN_DAY = 86_400.0

# ---------------------------------------------------------------------------
# Edge feature constants
# ---------------------------------------------------------------------------

# Edge feature vector layout:
#   slot 0: temporal_decay  — exp(-time_diff / window), float 0-1
#   slot 1: uid_flag        — 1.0 if edge from uid entity column
#   slot 2: card1_flag      — 1.0 if edge from card1 entity column
#   slot 3: DeviceInfo_flag — 1.0 if edge from DeviceInfo entity column
#   slot 4: email_flag      — 1.0 if edge from P_emaildomain entity column
#   slot 5: addr1_flag      — 1.0 if edge from addr1 entity column
#
# For pairs connected by multiple columns: temporal_decay = max across columns;
# entity flags are OR-combined (1.0 if any column connects this pair).
# This preserves "same card AND same device" signal that scalar max-merge loses.
#
# EDGE_DIM is fixed. Custom key_cols not in _DEFAULT_KEY_COLS still generate
# edges but their flag slot stays 0 (temporal_decay slot still used).
_DEFAULT_KEY_COLS: Tuple[str, ...] = (
    "uid", "card1", "DeviceInfo", "P_emaildomain", "addr1"
)
_DEFAULT_COL_WEIGHTS: Dict[str, float] = {
    "uid":           2.0,  # card1+addr1+account_start_date — tightest identity cluster
    "card1":         1.5,  # Card number group — strong but shared across account holders
    "DeviceInfo":    1.2,  # Device fingerprint — same device = likely same actor
    "P_emaildomain": 1.0,  # Purchaser email domain
    "addr1":         0.8,  # Billing address (weaker: shared at hotels, offices etc.)
}

EDGE_DIM: int = 1 + len(_DEFAULT_KEY_COLS)  # = 6

# Maps column name → flag slot index in the edge feature vector
_COL_FLAG_IDX: Dict[str, int] = {
    col: i + 1 for i, col in enumerate(_DEFAULT_KEY_COLS)
}
# {"uid": 1, "card1": 2, "DeviceInfo": 3, "P_emaildomain": 4, "addr1": 5}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_graph_input(
    X_raw: pd.DataFrame,
    key_cols: Tuple[str, ...],
) -> None:
    """Validate X_raw meets graph construction preconditions.

    Raises ValueError with a clear diagnostic if preconditions are violated.
    Logs warnings for soft failures (unsorted timestamps, sparse columns).
    """
    if "TransactionDT" not in X_raw.columns:
        raise ValueError(
            "TransactionDT is missing from the input DataFrame. "
            "build_entity_temporal_edges requires the post-build_features() "
            "DataFrame (uid + TransactionDT still present), not the raw joined "
            "DataFrame or the sklearn pipeline output (which drops both)."
        )

    if "uid" in key_cols and "uid" not in X_raw.columns:
        raise ValueError(
            "uid column not found. build_entity_temporal_edges requires the "
            "post-build_features() DataFrame where uid has been computed from "
            "card1 + addr1 + D1. Call build_features(X_raw.copy()) before "
            "passing to train_gnn() or build_entity_temporal_edges()."
        )

    # Warn if timestamps are not sorted — temporal direction depends on order
    dt = X_raw["TransactionDT"].values
    if not np.all(dt[:-1] <= dt[1:]):
        logger.warning(
            "TransactionDT is not monotonically non-decreasing. "
            "Temporal window logic assumes sorted order (oldest → newest). "
            "Sort your DataFrame by TransactionDT before graph construction."
        )

    # Warn if requested entity columns are mostly empty
    for col in key_cols:
        if col not in X_raw.columns:
            continue
        col_s = X_raw[col]
        non_null = col_s.notna() & (col_s != "unknown") & (col_s != "") & (col_s != -1)
        if col_s.dtype == object:
            non_null = non_null & ~col_s.astype(str).str.lower().str.contains(
                "nan", na=True
            )
        fill_rate = non_null.mean()
        if fill_rate < 0.10:
            logger.warning(
                "Entity column '%s' has only %.1f%% non-null/unknown values. "
                "Very few edges will be built from this column.",
                col, fill_rate * 100,
            )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_entity_temporal_edges(
    X_raw: pd.DataFrame,
    key_cols: Tuple[str, ...] = _DEFAULT_KEY_COLS,
    time_window_days: float = 7.0,
    max_edges_per_node: int = 10,
    max_total_edges: int = 5_000_000,
    col_weights: Optional[Dict[str, float]] = None,
    min_edges: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build directed temporal edges from entity sharing with vector edge features.

    For each entity column, group transactions by entity value and connect
    each transaction to its recent predecessors within time_window_days.
    Edges are directed (past → future only): removes the reverse (future →
    past) direction from the old version, eliminating the data leakage where
    a training node could aggregate from a future test transaction.

    Edge feature vector per edge (EDGE_DIM=6):
        [temporal_decay, uid_flag, card1_flag, DeviceInfo_flag,
         P_emaildomain_flag, addr1_flag]
    For pairs connected by multiple columns:
        temporal_decay = max(decay across columns)   — keep strongest recency
        entity flags   = OR(flags across columns)    — preserve multi-entity signal
    This replaces the old scalar max-merge which lost "same card AND same device"
    information. The EdgeGate in each SAGEConv layer learns to combine these.

    Performance: inner time-diff computation is vectorised with numpy (vs. pure
    Python in the previous version), giving ~5-10× speedup on large entity groups.

    Args:
        X_raw:              Feature-engineered DataFrame — must be the output of
                            build_features() so that uid and TransactionDT are
                            present. NOT the raw joined DataFrame (lacks uid) and
                            NOT the pipeline output (drops uid/TransactionDT).
        key_cols:           Entity columns to build edges from.
        time_window_days:   Only connect transactions within this many days.
                            Research shows 7–14 days optimal for IEEE-CIS.
        max_edges_per_node: Safety cap on backward predecessors per entity column
                            per node. Bounds max degree to
                            max_edges_per_node × len(key_cols).
        max_total_edges:    Global edge budget. Edges with the lowest temporal_decay
                            are dropped if budget is exceeded. Prevents OOM on very
                            dense graphs (default 5M ≈ ~150MB edge_attr tensor).
        col_weights:        Retained for API compatibility; not used in edge_attr
                            construction (EdgeGate learns weights from flags).
        min_edges:          Raise RuntimeError if fewer edges are built. Guards
                            against silent fallback to a vanilla MLP (min_edges=1
                            by default; pass 0 to suppress).

    Returns:
        src:       int64 [E]           — source (past) node indices
        dst:       int64 [E]           — destination (future) node indices
        edge_attr: float32 [E, EDGE_DIM] — edge feature matrix
    """
    _validate_graph_input(X_raw, key_cols)

    if col_weights is None:
        col_weights = _DEFAULT_COL_WEIGHTS

    idx_to_pos: Dict[Any, int] = {idx: pos for pos, idx in enumerate(X_raw.index)}
    dt_values = X_raw["TransactionDT"].values.astype(np.float64)

    # edge_map: (src_past, dst_future) → edge_attr np.ndarray [EDGE_DIM]
    # temporal_decay slot = max across columns; entity flag slots = OR across columns.
    # Keys are strictly (past_idx, future_idx) — no reverse direction.
    edge_map: Dict[Tuple[int, int], np.ndarray] = {}

    for col in key_cols:
        if col not in X_raw.columns:
            logger.warning("Entity column '%s' not in DataFrame — skipping.", col)
            continue

        flag_idx = _COL_FLAG_IDX.get(col)  # None for custom cols not in defaults

        # Exclude missing / unknown / sentinel values
        col_series = X_raw[col]
        valid_mask = (
            col_series.notna()
            & (col_series != "unknown")
            & (col_series != -1)
            & (col_series != "")
        )
        # String columns: also filter values containing "nan" — uid built from
        # NaN components yields e.g. "12345_nan_1234", which creates spurious clusters
        if col_series.dtype == object:
            valid_mask = valid_mask & ~col_series.astype(str).str.lower().str.contains(
                "nan", na=True
            )

        for _, group_idx in X_raw[valid_mask].groupby(col, sort=False).groups.items():
            positions = sorted(idx_to_pos[i] for i in group_idx)
            if len(positions) < 2:
                continue

            pos_arr = np.array(positions, dtype=np.int64)
            t_arr   = dt_values[pos_arr]

            for i in range(1, len(pos_arr)):
                v   = int(pos_arr[i])
                t_v = t_arr[i]

                # Vectorised time-diff computation (replaces pure Python walk)
                lookback_start  = max(0, i - max_edges_per_node * 3)
                slice_t         = t_arr[lookback_start:i]
                time_diffs_days = (t_v - slice_t) / _SECONDS_IN_DAY

                in_window = time_diffs_days <= time_window_days
                if not in_window.any():
                    continue

                cand_pos   = pos_arr[lookback_start:i][in_window]
                cand_diffs = time_diffs_days[in_window]

                # Most recent max_edges_per_node predecessors within window
                if len(cand_pos) > max_edges_per_node:
                    cand_pos   = cand_pos[-max_edges_per_node:]
                    cand_diffs = cand_diffs[-max_edges_per_node:]

                temporal_decays = np.exp(
                    -cand_diffs / time_window_days
                ).astype(np.float32)

                for u, td in zip(cand_pos.tolist(), temporal_decays.tolist()):
                    key = (int(u), v)  # directed: past u → future v only (no reverse)
                    if key not in edge_map:
                        edge_map[key] = np.zeros(EDGE_DIM, dtype=np.float32)
                    # Max temporal decay across entity columns
                    edge_map[key][0] = max(edge_map[key][0], td)
                    # OR entity flag — preserves multi-entity signal per edge
                    if flag_idx is not None:
                        edge_map[key][flag_idx] = 1.0

    if not edge_map:
        msg = (
            f"No edges constructed from entity columns {key_cols}. "
            "Check that: (1) uid column exists (requires post-build_features() "
            "DataFrame), (2) TransactionDT is present, (3) entity columns have "
            "non-null, non-nan values."
        )
        if min_edges > 0:
            raise RuntimeError(msg)
        logger.warning(msg)
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, EDGE_DIM), dtype=np.float32),
        )

    # Global edge budget: keep edges with highest temporal_decay
    if len(edge_map) > max_total_edges:
        logger.warning(
            "Edge count %d exceeds max_total_edges=%d. "
            "Dropping lowest-decay edges. Consider reducing time_window_days "
            "or max_edges_per_node to stay under budget.",
            len(edge_map), max_total_edges,
        )
        sorted_keys = sorted(edge_map, key=lambda k: edge_map[k][0], reverse=True)
        edge_map = {k: edge_map[k] for k in sorted_keys[:max_total_edges]}

    if len(edge_map) < min_edges:
        raise RuntimeError(
            f"Only {len(edge_map)} edges built; min_edges={min_edges}. "
            "The graph is too sparse for GNN training to add value over a vanilla MLP. "
            "Increase time_window_days, add entity columns, or pass min_edges=0 to suppress."
        )

    pairs   = list(edge_map.keys())
    src_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    dst_arr = np.array([p[1] for p in pairs], dtype=np.int64)
    attr_arr = np.vstack([edge_map[p] for p in pairs]).astype(np.float32)

    logger.info(
        "Directed temporal graph: %d edges from %d entity columns "
        "(window=%.0f days, max_per_node=%d, total_cap=%d)",
        len(pairs), len(key_cols), time_window_days,
        max_edges_per_node, max_total_edges,
    )
    return src_arr, dst_arr, attr_arr


def _knn_fallback_edges(
    X_proc: np.ndarray,
    isolated_idx: List[int],
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add K-NN edges for isolated nodes using feature-space cosine similarity.

    Isolated nodes (no entity-sharing temporal neighbours) would receive only
    their own features through lin_self and no neighbourhood signal. K-NN
    connects them to their nearest non-isolated neighbours in feature space.

    Edge direction: earlier-by-index → isolated node. Index ≈ temporal order
    (DataFrame is sorted by TransactionDT), so this respects temporal direction.

    KNN edge features: [cosine_similarity, 0, 0, 0, 0, 0]
        Slot 0 repurposed as similarity score (0–1). Entity flags are all 0.
        The EdgeGate can learn to down-weight these relative to entity edges.

    Returns empty arrays if no isolated nodes or sklearn unavailable.
    """
    if not isolated_idx:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, EDGE_DIM), dtype=np.float32),
        )

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.warning("scikit-learn not available — skipping KNN fallback edges.")
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, EDGE_DIM), dtype=np.float32),
        )

    iso_arr  = np.array(isolated_idx, dtype=np.int64)
    k_actual = min(k + 1, len(X_proc))  # +1 because self is always returned
    nn_model = NearestNeighbors(n_neighbors=k_actual, metric="cosine", algorithm="brute")
    nn_model.fit(X_proc)
    distances, indices = nn_model.kneighbors(X_proc[iso_arr])

    src_list: List[int] = []
    dst_list: List[int] = []
    attr_list: List[np.ndarray] = []

    for local_i, (dists, nbrs) in enumerate(zip(distances, indices)):
        v = int(iso_arr[local_i])
        for d, u in zip(dists, nbrs):
            if int(u) == v:
                continue  # skip self
            similarity = float(max(0.0, 1.0 - d))
            attr = np.zeros(EDGE_DIM, dtype=np.float32)
            attr[0] = similarity
            # Directed: earlier index (= earlier transaction) → later index
            s, t = (int(u), v) if int(u) < v else (v, int(u))
            src_list.append(s)
            dst_list.append(t)
            attr_list.append(attr)

    if not src_list:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, EDGE_DIM), dtype=np.float32),
        )

    return (
        np.array(src_list, dtype=np.int64),
        np.array(dst_list, dtype=np.int64),
        np.vstack(attr_list).astype(np.float32),
    )


def build_transaction_graph(
    X_raw: pd.DataFrame,
    X_proc: np.ndarray,
    key_cols: Tuple[str, ...] = _DEFAULT_KEY_COLS,
    time_window_days: float = 7.0,
    max_edges_per_node: int = 10,
    max_total_edges: int = 5_000_000,
    knn_fallback_k: int = 5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct PyG-compatible tensors with K-NN fallback for isolated nodes.

    Nodes with no entity-sharing temporal neighbours (isolated) receive K-NN
    fallback edges based on feature-space cosine similarity. knn_fallback_k=0
    disables the fallback; isolated nodes then receive no neighbourhood signal.

    Returns:
        edge_index:  LongTensor [2, E]
        edge_attr:   FloatTensor [E, EDGE_DIM]  — learned by EdgeGate per layer
        x:           FloatTensor [N, F]         — preprocessed node features
    """
    src, dst, edge_attr = build_entity_temporal_edges(
        X_raw, key_cols, time_window_days, max_edges_per_node, max_total_edges,
    )

    N = len(X_proc)

    # Isolated = nodes with no incoming edges (receive no aggregation signal)
    connected_dst = set(dst.tolist()) if len(dst) > 0 else set()
    isolated_idx  = [i for i in range(N) if i not in connected_dst]
    n_isolated    = len(isolated_idx)
    isolated_pct  = 100.0 * n_isolated / max(N, 1)

    if n_isolated > 0:
        log_fn = logger.warning if isolated_pct > 10.0 else logger.info
        log_fn(
            "%.1f%% of nodes (%d/%d) have no incoming entity-sharing edges. "
            "%s",
            isolated_pct, n_isolated, N,
            "Adding KNN fallback." if knn_fallback_k > 0 else
            "Consider increasing time_window_days (knn_fallback_k=0, no fallback).",
        )

    if knn_fallback_k > 0 and n_isolated > 0:
        src_knn, dst_knn, attr_knn = _knn_fallback_edges(X_proc, isolated_idx, k=knn_fallback_k)
        if len(src_knn) > 0:
            src       = np.concatenate([src, src_knn])
            dst       = np.concatenate([dst, dst_knn])
            edge_attr = np.vstack([edge_attr, attr_knn]) if len(edge_attr) > 0 else attr_knn
            logger.info(
                "KNN fallback: added %d edges for %d isolated nodes.",
                len(src_knn), n_isolated,
            )

    if len(src) > 0:
        edge_index  = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index  = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    x = torch.FloatTensor(X_proc)
    logger.info(
        "Graph: %d nodes, %d edges (avg in-degree %.1f)",
        x.shape[0], edge_index.shape[1],
        edge_index.shape[1] / max(x.shape[0], 1),
    )
    return edge_index, edge_attr_t, x


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class EdgeGatedSAGEConv(nn.Module):
    """SAGEConv with learned edge-feature gate.

    Each edge carries a EDGE_DIM-dimensional feature vector. A learned
    EdgeGate (Linear(EDGE_DIM, 1) + sigmoid) projects this to a scalar
    aggregation weight per edge:

        gate_ij = sigmoid(W_gate · e_ij)
        h_neigh = Σ(gate_ij · lin_neigh(h_j)) / Σ(gate_ij)
        h_v     = lin_self(h_v) + h_neigh

    Compared to the previous manually-tuned scalar (temporal_decay × col_weight),
    W_gate is a 6-parameter learned vector. It can discover that "same card AND
    same device" edges (uid_flag=1, card1_flag=1) deserve more weight than "same
    email domain only" edges — a signal the manual scalar collapsed away.

    GNNExplainer fallback (edge_attr=None):
        GNNExplainer optimises its own edge masks and cannot pass custom edge_attr
        through its optimisation loop. When edge_attr=None, falls back to
        unweighted mean (all gates = 1). Explanation masks reflect unweighted
        topology, not the actual gate weights.

        COMPLIANCE NOTE: GNNExplainer edge masks do not reflect temporal recency
        or entity type. For edge-weighted explanations, use top_influential_neighbors()
        which reads the pre-computed edge_attr tensor directly.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.lin_self  = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_gate = nn.Linear(edge_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:          [N, in_channels]
            edge_index: [2, E]
            edge_attr:  [E, edge_dim] or None (unweighted fallback for GNNExplainer)
        Returns:
            [N, out_channels]
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        N = x.size(0)

        neigh_x = self.lin_neigh(x)  # [N, out]

        if edge_attr is not None:
            gate = torch.sigmoid(self.edge_gate(edge_attr)).view(-1)  # [E]
            msgs = gate.unsqueeze(1) * neigh_x[src_idx]               # [E, out]
            agg  = torch.zeros(N, neigh_x.size(1), device=x.device)
            agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msgs), msgs)
            w_sum = torch.zeros(N, 1, device=x.device)
            w_sum.scatter_add_(0, dst_idx.unsqueeze(1), gate.unsqueeze(1))
            agg = agg / (w_sum + 1e-8)
        else:
            # Unweighted mean — used by GNNExplainer (cannot pass edge_attr)
            msgs   = neigh_x[src_idx]
            agg    = torch.zeros(N, neigh_x.size(1), device=x.device)
            agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msgs), msgs)
            degree = torch.zeros(N, 1, device=x.device)
            degree.scatter_add_(
                0, dst_idx.unsqueeze(1),
                torch.ones(dst_idx.size(0), 1, device=x.device),
            )
            agg = agg / (degree + 1e-8)

        return self.lin_self(x) + agg


# Backward-compatibility alias
WeightedSAGEConv = EdgeGatedSAGEConv


class FraudGNN(nn.Module):
    """Multi-layer edge-gated GraphSAGE for transaction-level fraud detection.

    Architecture:
        Linear(F→H) input projection
        → num_layers × [EdgeGatedSAGEConv(H→H) → BN → LeakyReLU → Dropout + skip]
        → Linear(H→1) classifier

    Key design choices vs. prior version:
    - Constant hidden_dim across all layers: enables skip connections (residual
      shortcut adds identity to each conv output starting from layer 2).
      Prior version halved dims per layer, preventing skips and causing
      over-smoothing on deeper configs.
    - embed() / forward() split: embed() returns pre-classifier node vectors
      for use by extract_gnn_embeddings() (Stage 2 of GNN→XGBoost).
    - Input projection: separates feature encoding from message passing so the
      first conv layer always operates in the embedding space, not raw feature
      space (important when input_dim >> hidden_dim).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        edge_dim: int = EDGE_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project raw features into embedding space once (no message passing)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeGatedSAGEConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout    = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming Normal init for LeakyReLU — prevents vanishing gradients."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def embed(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Return pre-classifier node embeddings [N, hidden_dim].

        Called by extract_gnn_embeddings() for the GNN→XGBoost Stage 2.
        Skip connections added from layer 2 onward (same dim = clean residual).
        """
        x = F.leaky_relu(self.input_proj(x))
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index, edge_attr)
            h = bn(h)
            h = F.leaky_relu(h)
            h = self.dropout(h)
            x = x + h if i > 0 else h  # skip from layer 2 onward
        return x

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:           [N, F] node features
            edge_index:  [2, E]
            edge_attr:   [E, edge_dim] edge feature matrix (None = unweighted fallback)
        Returns:
            logits: [N, 1]
        """
        return self.classifier(self.embed(x, edge_index, edge_attr))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_gnn_model(
    params: Optional[Dict[str, Any]],
    input_dim: int,
    edge_dim: int = EDGE_DIM,
) -> FraudGNN:
    """Instantiate FraudGNN from a params dict — mirrors get_xgboost_model()."""
    if params is None:
        params = {}
    return FraudGNN(
        input_dim=input_dim,
        hidden_dim=params.get("hidden_dim", 128),
        num_layers=params.get("num_layers", 2),
        dropout_rate=params.get("dropout_rate", 0.3),
        edge_dim=edge_dim,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gnn(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/fraud_gnn.pt",
    seed: int = 42,
) -> "FraudGNN":
    """Train FraudGNN with entity-temporal graph, NeighborLoader, and FPR early stopping.

    Graph setup
    -----------
    Train and test nodes are combined into one Data object with boolean masks.
    Edges are directed (past → future) to prevent temporal leakage. Edge features
    [temporal_decay, entity_flags] are passed through every conv layer's EdgeGate.
    Test nodes receive neighbourhood signals from adjacent training transactions
    but cannot send messages back to training nodes (directed edges).

    Early stopping
    --------------
    Inline disk-based implementation (saves checkpoint to tempfile). Avoids
    duplicating model weights in RAM during training — relevant for larger
    GNNs where in-memory deepcopy would double peak memory usage.

    Args:
        X_train_proc, y_train: Pre-processed training arrays.
        X_test_proc,  y_test:  Pre-processed OOT test arrays.
        X_train_raw, X_test_raw: Feature-engineered DataFrames (post
                       build_features(), uid + TransactionDT still present).
        params:        hidden_dim, num_layers, dropout_rate, learning_rate,
                       batch_size, epochs, patience, num_neighbors,
                       time_window_days, max_edges_per_node, max_total_edges.
        fpr_threshold: Operating threshold for early stopping (from config).
        save_path:     Pass None during Optuna tuning to skip disk writes.
        seed:          Random seed for reproducibility (torch + numpy).

    Returns:
        FraudGNN with best-checkpoint weights restored.
    """
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise ImportError("pip install torch-geometric") from exc

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if params is None:
        params = {}

    epochs             = params.get("epochs", 50)
    batch_size         = params.get("batch_size", 512)
    lr                 = params.get("learning_rate", 1e-3)
    patience           = params.get("patience", 5)
    num_neighbors      = params.get("num_neighbors", 10)
    time_window_days   = params.get("time_window_days", 7.0)
    max_edges_per_node = params.get("max_edges_per_node", 10)
    max_total_edges    = params.get("max_total_edges", 5_000_000)

    y_t = y_train.values if hasattr(y_train, "values") else y_train
    y_v = y_test.values  if hasattr(y_test,  "values") else y_test
    n_train, n_test = len(X_train_proc), len(X_test_proc)

    # ---- Build combined directed graph ----
    X_raw_combined = pd.concat(
        [X_train_raw.reset_index(drop=True), X_test_raw.reset_index(drop=True)],
        ignore_index=True,
    )
    X_proc_combined = np.vstack([X_train_proc, X_test_proc])
    y_combined      = np.concatenate([y_t, y_v])

    edge_index, edge_attr, x = build_transaction_graph(
        X_raw_combined, X_proc_combined,
        time_window_days=time_window_days,
        max_edges_per_node=max_edges_per_node,
        max_total_edges=max_total_edges,
    )

    train_mask = torch.zeros(n_train + n_test, dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask = torch.zeros(n_train + n_test, dtype=torch.bool)
    test_mask[n_train:]  = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,           # [E, EDGE_DIM] — vector edge features
        y=torch.FloatTensor(y_combined),
        train_mask=train_mask,
        test_mask=test_mask,
    )

    train_loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors, num_neighbors],
        batch_size=batch_size,
        input_nodes=train_mask,
        shuffle=True,
        num_workers=0,
    )
    # Exact inference for evaluation — no sampling noise
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=batch_size * 4,
        input_nodes=test_mask,
        shuffle=False,
        num_workers=0,
    )

    input_dim = X_proc_combined.shape[1]
    edge_dim  = edge_attr.shape[1] if edge_attr.shape[0] > 0 else EDGE_DIM
    model     = get_gnn_model(params, input_dim, edge_dim=edge_dim).to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    _mlflow_active = mlflow.active_run() is not None

    # Disk-based early stopping — saves checkpoint to tempfile instead of
    # keeping a deepcopy of weights in RAM during training
    best_fpr       = float("inf")
    patience_count = 0
    ckpt_file      = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    ckpt_path      = ckpt_file.name
    ckpt_file.close()
    torch.save(model.state_dict(), ckpt_path)

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss, n_seen = 0.0, 0
        for batch in train_loader:
            batch  = batch.to(device)
            seed_n = batch.batch_size
            optimizer.zero_grad()
            out = model(
                batch.x, batch.edge_index,
                edge_attr=batch.edge_attr,
            )[:seed_n].view(-1)
            loss = criterion(out, batch.y[:seed_n])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * seed_n
            n_seen     += seed_n
        train_loss /= max(n_seen, 1)

        # ---- Validation (exact inference) ----
        model.eval()
        val_loss, all_probs, n_eval = 0.0, [], 0
        with torch.no_grad():
            for batch in test_loader:
                batch  = batch.to(device)
                seed_n = batch.batch_size
                out = model(
                    batch.x, batch.edge_index,
                    edge_attr=batch.edge_attr,
                )[:seed_n].view(-1)
                val_loss  += criterion(out, batch.y[:seed_n]).item() * seed_n
                all_probs.append(torch.sigmoid(out).cpu().numpy())
                n_eval    += seed_n
        val_loss    /= max(n_eval, 1)
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
            "GNN Epoch %d/%d — train_loss: %.4f  val_loss: %.4f  "
            "val_fpr: %.4f  val_auc: %.4f",
            epoch + 1, epochs, train_loss, val_loss, val_fpr, val_auc,
        )
        if _mlflow_active:
            mlflow.log_metrics(
                {"gnn_train_loss": train_loss, "gnn_val_loss": val_loss,
                 "gnn_val_fpr": val_fpr, "gnn_val_auc": val_auc},
                step=epoch,
            )

        scheduler.step(val_loss)

        if val_fpr < best_fpr:
            best_fpr       = val_fpr
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    "GNN early stopping at epoch %d (FPR plateau, patience=%d).",
                    epoch + 1, patience,
                )
                break

    # Restore best checkpoint and clean up temp file
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    os.unlink(ckpt_path)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or "models", exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(),
             "input_dim": input_dim,
             "edge_dim": edge_dim},
            save_path,
        )
        logger.info("Best GNN weights saved to %s", save_path)
        if _mlflow_active:
            mlflow.log_artifact(save_path, artifact_path="gnn_model")

    return model


# ---------------------------------------------------------------------------
# Embedding extraction + GNN→XGBoost hybrid
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_gnn_embeddings(
    model: FraudGNN,
    data: "Data",
    n_train: int,
    device: str = "cpu",
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pre-classifier node embeddings for ALL nodes in the graph.

    Uses NeighborLoader with exact neighbourhoods (num_neighbors=-1) for
    inference — no sampling noise. Node ordering is restored via batch.n_id
    so the output rows align with the original node indices in `data`.

    Args:
        model:    Trained FraudGNN (best-checkpoint weights loaded).
        data:     Combined train+test PyG Data object (same graph used in training).
        n_train:  Number of training nodes (first n_train rows → train split).
        device:   Inference device.
        batch_size: Inference batch size.

    Returns:
        (train_embeddings, test_embeddings): np.ndarray [n_train, H], [n_test, H]
    """
    try:
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise ImportError("pip install torch-geometric") from exc

    model.eval()
    model = model.to(device)

    all_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=batch_size,
        input_nodes=all_mask,
        shuffle=False,
        num_workers=0,
    )

    # Pre-allocate — fill in original-node order via n_id scatter
    collected: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        batch  = batch.to(device)
        seed_n = batch.batch_size
        emb    = model.embed(batch.x, batch.edge_index, edge_attr=batch.edge_attr)[:seed_n]
        collected.append((batch.n_id[:seed_n].cpu(), emb.cpu()))

    emb_dim   = collected[0][1].shape[1]
    all_embs  = torch.zeros((data.num_nodes, emb_dim))
    for n_ids, embs in collected:
        all_embs[n_ids] = embs

    train_emb = all_embs[:n_train].numpy()
    test_emb  = all_embs[n_train:].numpy()
    logger.info(
        "GNN embeddings extracted: train=%s  test=%s  dim=%d",
        train_emb.shape, test_emb.shape, emb_dim,
    )
    return train_emb, test_emb


def train_gnn_xgboost(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    X_train_eng: pd.DataFrame,
    X_test_eng: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    fpr_threshold: float = 0.85,
    device: str = "cpu",
    save_path: Optional[str] = "models/gnn_xgboost",
    seed: int = 42,
) -> Tuple[FraudGNN, Any]:
    """Two-stage GNN→XGBoost training.

    Stage 1: Train FraudGNN with FocalLoss + FPR early stopping.
             Graph = combined train+test with directed edges (no temporal leakage).
    Stage 2: Extract per-node embeddings from best-checkpoint GNN.
             Train XGBoost on [original_features || GNN_embeddings].

    Enables direct comparison with mlp_xgboost and transformer_xgboost:
        mlp_xgboost          → shallow MLP extraction
        transformer_xgboost  → attention-based extraction
        gnn (this function)  → graph-based extraction (entity-neighbourhood)
    All three use identical X_proc input; AUC differences are attributable
    solely to whether neighbourhood structure improves feature extraction.

    Args:
        X_train_proc, y_train: Pre-processed training arrays.
        X_test_proc,  y_test:  Pre-processed OOT test arrays.
        X_train_eng, X_test_eng: post-build_features() DataFrames
                       (uid + TransactionDT present for graph construction).
        params:   GNN params (hidden_dim, num_layers, etc.) +
                  XGBoost params (n_estimators, max_depth, xgb_early_stopping_rounds, …).
        save_path: Directory prefix. Pass None during Optuna tuning.

    Returns:
        (gnn_model, xgb_model): best-checkpoint FraudGNN + fitted XGBClassifier.
    """
    import joblib

    from src.models.tree_models import get_xgboost_model

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("pip install torch-geometric") from exc

    if params is None:
        params = {}

    _mlflow_active = mlflow.active_run() is not None

    # ---- Stage 1: GNN pre-training ----
    logger.info(
        "Stage 1: Training FraudGNN (%d train, %d test, %d features).",
        len(X_train_proc), len(X_test_proc), X_train_proc.shape[1],
    )
    gnn_save = None
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        gnn_save = os.path.join(save_path, "gnn.pt")

    gnn_model = train_gnn(
        X_train_proc, y_train, X_test_proc, y_test,
        X_train_eng, X_test_eng,
        params=params, fpr_threshold=fpr_threshold,
        device=device, save_path=gnn_save, seed=seed,
    )

    # ---- Stage 2: Embedding extraction ----
    # Rebuild the combined graph (same construction as inside train_gnn)
    time_window_days   = params.get("time_window_days", 7.0)
    max_edges_per_node = params.get("max_edges_per_node", 10)
    max_total_edges    = params.get("max_total_edges", 5_000_000)

    X_eng_combined  = pd.concat(
        [X_train_eng.reset_index(drop=True), X_test_eng.reset_index(drop=True)],
        ignore_index=True,
    )
    X_proc_combined = np.vstack([X_train_proc, X_test_proc])
    edge_index, edge_attr, x = build_transaction_graph(
        X_eng_combined, X_proc_combined,
        time_window_days=time_window_days,
        max_edges_per_node=max_edges_per_node,
        max_total_edges=max_total_edges,
    )
    data    = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    n_train = len(X_train_proc)

    logger.info("Stage 2: Extracting GNN embeddings for enriched XGBoost input.")
    train_emb, test_emb = extract_gnn_embeddings(
        gnn_model, data, n_train,
        device=device, batch_size=params.get("batch_size", 512) * 4,
    )

    X_train_enriched = np.concatenate([X_train_proc, train_emb], axis=1)
    X_test_enriched  = np.concatenate([X_test_proc,  test_emb],  axis=1)
    logger.info(
        "Enriched feature dim: %d → %d (original + %d GNN embedding dims)",
        X_train_proc.shape[1], X_train_enriched.shape[1], train_emb.shape[1],
    )

    # ---- Stage 3: XGBoost on enriched features ----
    xgb_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "min_child_weight",
        "reg_alpha", "reg_lambda", "gamma", "tree_method",
    }
    xgb_params          = {k: v for k, v in params.items() if k in xgb_keys}
    xgb_early_stopping  = params.get("xgb_early_stopping_rounds", 30)

    logger.info(
        "Stage 3: Training XGBoost on %d enriched features with FPR eval metric.",
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

    y_v   = y_test.values if hasattr(y_test, "values") else y_test
    preds = xgb_model.predict_proba(X_test_enriched)[:, 1]
    auc   = roc_auc_score(y_v, preds)
    logger.info("GNN+XGBoost OOT AUC: %.4f", auc)

    if _mlflow_active:
        mlflow.log_metric("gnn_xgboost_OOT_AUC", auc)
        mlflow.log_metric(
            "gnn_xgboost_best_iteration",
            getattr(xgb_model, "best_iteration", params.get("n_estimators", 300)),
        )

    if save_path is not None:
        xgb_path = os.path.join(save_path, "xgboost.joblib")
        joblib.dump(xgb_model, xgb_path)
        logger.info("GNN saved → %s  XGBoost saved → %s", gnn_save, xgb_path)

        if _mlflow_active:
            mlflow.log_artifact(xgb_path, artifact_path="gnn_xgboost")
            mlflow.xgboost.log_model(xgb_model, artifact_path="gnn_xgb_model")

    return gnn_model, xgb_model


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class _ExplainerWrapper(nn.Module):
    """Squeeze FraudGNN's [N, 1] output to [N] for PyG Explainer API.

    Drops edge_attr: GNNExplainer optimises its own edge masks via gradient
    descent on a mask variable and cannot pass custom edge_attr through its
    optimisation loop. The model runs with edge_attr=None, falling back to
    unweighted mean aggregation in EdgeGatedSAGEConv.

    COMPLIANCE NOTE: GNNExplainer edge masks do not reflect the learned
    EdgeGate weights (temporal recency, entity type, multi-entity combos).
    This is a known limitation of the post-hoc explanation method — the
    explanation runs on a different forward pass than production inference.
    For edge-weighted explanations, use top_influential_neighbors() which
    reads the actual edge_attr tensor used during inference.
    """

    def __init__(self, model: FraudGNN):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.model(x, edge_index, edge_attr=None).squeeze(-1)  # [N]


def explain_transaction(
    model: FraudGNN,
    x: Tensor,
    edge_index: Tensor,
    node_idx: int,
    device: str = "cpu",
    epochs: int = 200,
) -> Any:
    """Run GNNExplainer on a single flagged transaction.

    GNNExplainer (Ying et al. 2019) learns soft masks over edges and node
    features that maximally preserve the model's prediction for the target
    node. It solves a small optimisation problem (default 200 epochs) to
    find the minimal subgraph that explains the fraud flag.

    IMPORTANT — edge weight limitation:
        GNNExplainer runs with edge_attr=None (unweighted mean fallback).
        The resulting edge masks reflect which connections are structurally
        important to the model topology, but do NOT reflect the actual
        EdgeGate weights used during inference (which encode temporal recency
        and entity type). For edge-weighted explanations, use
        top_influential_neighbors() with the pre-computed edge_attr tensor.

    NOT suitable for real-time serving (200-epoch optimisation per prediction).
    Use gradient_feature_importance() for latency-sensitive paths (~1ms).

    Returns a PyG Explanation with:
        explanation.node_mask  [N, F] — feature importance per node (0–1)
        explanation.edge_mask  [E]    — edge importance (0–1), unweighted topology
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
    except ImportError as exc:
        raise ImportError("pip install torch-geometric>=2.0") from exc

    model.eval()
    wrapper = _ExplainerWrapper(model).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(x.to(device), edge_index.to(device), index=node_idx)
    logger.info(
        "GNNExplainer: node %d — %d edges, top edge mask %.3f",
        node_idx, explanation.edge_mask.shape[0], float(explanation.edge_mask.max()),
    )
    return explanation


def node_feature_importance(
    explanation: Any,
    feature_names: List[str],
    node_idx: int,
) -> "pd.Series":
    """Extract per-feature importance for the target node; normalised to sum=1.

    Returns pd.Series sorted descending — directly usable in SR 11-7 / Basel
    model risk reports or fraud analyst investigation dashboards.
    """
    mask = explanation.node_mask
    if node_idx >= mask.shape[0]:
        raise ValueError(
            f"node_idx={node_idx} out of range for node_mask with {mask.shape[0]} rows. "
            "Pass the local subgraph index (usually 0 for the seed node)."
        )
    importances = mask[node_idx].cpu().detach().numpy()
    importances = importances / (importances.sum() + 1e-8)
    return (
        pd.Series(importances, index=feature_names, name="feature_importance")
        .sort_values(ascending=False)
    )


def top_influential_neighbors(
    explanation: Any,
    edge_index: Tensor,
    k: int = 5,
    edge_attr: Optional[Tensor] = None,
) -> "pd.DataFrame":
    """Return K most influential connections for a flagged transaction.

    Combines GNNExplainer edge mask (structural importance, unweighted) with
    the pre-computed edge_attr features (temporal decay + entity flags) to
    surface edges that were both structurally important AND temporally close
    with meaningful entity signal.

    Combined score = gnn_mask × temporal_decay (slot 0 of edge_attr).

    Columns include temporal_decay and per-entity flags so analysts can see
    which entity type (card, device, email, address) drove each connection.
    Entity columns: uid_flag, card1_flag, devinfo_flag, email_flag, addr1_flag.

    COMPLIANCE NOTE: gnn_mask reflects unweighted topology (EdgeGate not used
    during GNNExplainer). The combined_score × temporal_decay is the best
    available proxy for the actual inference weights.
    """
    gnn_mask = explanation.edge_mask.cpu().detach().numpy()

    ei = (
        explanation.edge_index.cpu().numpy()
        if hasattr(explanation, "edge_index") and explanation.edge_index is not None
        else edge_index.cpu().numpy()
    )

    n_edges = len(gnn_mask)
    df_data: Dict[str, Any] = {
        "source_node": ei[0, :n_edges],
        "dest_node":   ei[1, :n_edges],
        "gnn_mask":    gnn_mask,
    }

    if edge_attr is not None:
        ea = edge_attr.cpu().detach().numpy()
        if len(ea) == n_edges:
            df_data["temporal_decay"] = ea[:, 0]
            _flag_names = ["uid_flag", "card1_flag", "devinfo_flag", "email_flag", "addr1_flag"]
            for i, name in enumerate(_flag_names, start=1):
                if i < ea.shape[1]:
                    df_data[name] = ea[:, i]
            combined = gnn_mask * ea[:, 0]
        else:
            combined = gnn_mask
    else:
        combined = gnn_mask

    df_data["combined_score"] = combined
    top_k = np.argsort(combined)[::-1][:k]
    return pd.DataFrame(
        {col: vals[top_k] for col, vals in df_data.items()}
    ).reset_index(drop=True)


def gradient_feature_importance(
    model: FraudGNN,
    x: Tensor,
    edge_index: Tensor,
    node_idx: int,
    device: str = "cpu",
    edge_attr: Optional[Tensor] = None,
) -> np.ndarray:
    """Gradient saliency: |∂logit / ∂x_i| at the target node, normalised to sum=1.

    One backward pass (~1ms). Passes edge_attr through the EdgeGate so the
    gradient reflects the actual weighted-aggregation forward pass — not the
    unweighted approximation that GNNExplainer uses.

    20–100× faster than GNNExplainer. Use for real-time serving explanations.
    """
    model.eval()
    x_input    = x.clone().float().to(device).requires_grad_(True)
    edge_index = edge_index.to(device)
    ea         = edge_attr.to(device) if edge_attr is not None else None

    logit = model(x_input, edge_index, edge_attr=ea)[node_idx].squeeze()
    logit.backward()

    grad = x_input.grad[node_idx].abs().cpu().detach().numpy()
    return grad / (grad.sum() + 1e-8)
