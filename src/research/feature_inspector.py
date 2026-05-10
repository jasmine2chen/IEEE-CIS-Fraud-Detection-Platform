"""Feature inspection utilities for exploratory analysis and debugging.

Bridges the gap between raw data exploration and the production evaluation
harness in ``src/evaluation/``.  These tools are designed for interactive
use in notebooks and for producing the "short write-up" artefacts that
research engineers hand to stakeholders.

Usage
-----
    from src.research.feature_inspector import (
        feature_target_correlation,
        text_feature_audit,
        embedding_visualization,
    )
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# feature_target_correlation
# ---------------------------------------------------------------------------

def feature_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 30,
    train_X: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute point-biserial correlation of each numeric feature with a binary target.

    Also flags features where the train vs test KS p-value < 0.05 — a signal of
    potential temporal drift or leakage that warrants investigation.

    Args:
        X:       Feature DataFrame (test set, or full dataset).
        y:       Binary target series aligned with X.
        top_n:   Number of top features to return (by |correlation|).
        train_X: Optional training DataFrame for KS drift detection.
                 If provided, each feature's KS statistic and p-value are included.

    Returns:
        DataFrame with columns: feature, correlation, abs_corr,
        [ks_statistic, ks_pvalue, drift_flag] (if train_X provided).
        Sorted by abs_corr descending.
    """
    y_arr = y.values if hasattr(y, "values") else y
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in numeric_cols:
        x_col = X[col].values
        # Drop NaN pairs
        mask = ~np.isnan(x_col)
        if mask.sum() < 10:
            continue
        try:
            r, _ = scipy_stats.pointbiserialr(y_arr[mask], x_col[mask])
        except Exception:
            r = 0.0
        row: dict = {"feature": col, "correlation": round(float(r), 4), "abs_corr": round(abs(r), 4)}

        if train_X is not None and col in train_X.columns:
            try:
                ks_stat, ks_p = scipy_stats.ks_2samp(
                    train_X[col].dropna().values,
                    X[col].dropna().values,
                )
                row["ks_statistic"] = round(float(ks_stat), 4)
                row["ks_pvalue"]    = round(float(ks_p), 6)
                row["drift_flag"]   = ks_p < 0.05
            except Exception:
                row["ks_statistic"] = None
                row["ks_pvalue"]    = None
                row["drift_flag"]   = None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# text_feature_audit
# ---------------------------------------------------------------------------

def text_feature_audit(
    df: pd.DataFrame,
    text_cols: List[str],
    y: pd.Series,
    top_n_values: int = 20,
) -> Dict[str, pd.DataFrame]:
    """For each text column, show fraud rate and lift per unique value.

    Reveals which email domains, device strings, or product codes are the
    strongest fraud signals — *before* feature engineering.  Useful for
    justifying NLP or categorical feature design choices to stakeholders.

    Lift = (value fraud rate) / (overall fraud rate).  Values with lift > 2
    are strong signals; lift < 0.5 indicates a low-risk segment.

    Args:
        df:              DataFrame containing the text columns.
        text_cols:       List of column names to audit.
        y:               Binary fraud label series.
        top_n_values:    Number of top-lift values to return per column.

    Returns:
        Dict mapping column name → DataFrame with columns:
        value, count, fraud_count, fraud_rate, lift, pct_of_all_fraud.
    """
    y_arr = y.values if hasattr(y, "values") else y
    overall_fraud_rate = float(y_arr.mean())
    total_fraud = int(y_arr.sum())

    results: Dict[str, pd.DataFrame] = {}
    for col in text_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame — skipping.", col)
            continue

        col_data = df[col].fillna("(missing)").astype(str)
        rows = []
        for val in col_data.unique():
            mask = (col_data == val).values
            n = int(mask.sum())
            if n < 5:          # skip very rare values (unreliable fraud rate)
                continue
            n_fraud = int(y_arr[mask].sum())
            rate = n_fraud / n if n > 0 else 0.0
            lift = rate / overall_fraud_rate if overall_fraud_rate > 0 else 0.0
            pct_of_fraud = n_fraud / total_fraud if total_fraud > 0 else 0.0
            rows.append({
                "value":           val,
                "count":           n,
                "fraud_count":     n_fraud,
                "fraud_rate":      round(rate, 4),
                "lift":            round(lift, 2),
                "pct_of_all_fraud": round(pct_of_fraud, 4),
            })

        if not rows:
            continue

        result_df = (
            pd.DataFrame(rows)
            .sort_values("lift", ascending=False)
            .reset_index(drop=True)
            .head(top_n_values)
        )
        results[col] = result_df

    return results


# ---------------------------------------------------------------------------
# embedding_visualization
# ---------------------------------------------------------------------------

def embedding_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "umap",
    title: str = "Embedding Space",
    save_path: Optional[str] = None,
    sample_n: int = 5000,
) -> "plt.Figure":  # type: ignore[name-defined]
    """2D projection of encoder embeddings coloured by fraud label.

    Shows whether the encoder's representation space separates fraud from
    legitimate transactions — a useful sanity check after Stage 1 pre-training.

    Args:
        embeddings: Array of shape (N, d) — output of extract_mlp_embeddings().
        labels:     Binary labels (0=legit, 1=fraud) of shape (N,).
        method:     Projection method: "umap" (default) or "tsne".
                    Falls back to "tsne" if umap-learn is not installed.
        title:      Plot title.
        save_path:  If provided, save the figure to this path.
        sample_n:   Subsample to this many points for speed (t-SNE/UMAP are O(N²)).

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Subsample for speed
    n = len(embeddings)
    if n > sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_n, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Dimensionality reduction
    coords: Optional[np.ndarray] = None
    used_method = method

    if method == "umap":
        try:
            import umap as umap_lib  # type: ignore[import]
            reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed — falling back to t-SNE.")
            used_method = "tsne"

    if coords is None or used_method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)
        used_method = "tsne"

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.where(labels == 1, "#E84C4C", "#4C9BE8")
    alphas = np.where(labels == 1, 0.8, 0.3)

    for i in range(len(coords)):
        ax.scatter(coords[i, 0], coords[i, 1], c=colors[i], alpha=alphas[i],
                   s=6, linewidths=0)

    legit_patch = mpatches.Patch(color="#4C9BE8", label=f"Legitimate (n={int((labels==0).sum()):,})")
    fraud_patch = mpatches.Patch(color="#E84C4C", label=f"Fraud (n={int((labels==1).sum()):,})")
    ax.legend(handles=[legit_patch, fraud_patch], loc="upper right")
    ax.set_title(f"{title} [{used_method.upper()}]")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Embedding visualization saved to %s", save_path)

    return fig
