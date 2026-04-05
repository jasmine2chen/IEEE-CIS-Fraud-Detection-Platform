"""Fraud-specific evaluation plots.

Two functions cover the highest-value visualisations identified from
production fraud-detection systems:

1. plot_score_distributions — overlapping score histograms for fraud vs
   legitimate transactions, showing model separation at a glance.

2. plot_dollar_recall_curve — dollar recall and count recall vs FPR, so
   reviewers can see how much fraud value is caught at each operating point.
   This is more actionable than a ROC curve because it quantifies dollar
   impact rather than abstract TP/FP counts.

Both functions save to disk when ``save_path`` is provided and return the
``matplotlib.figure.Figure`` object for use in notebooks or MLflow logging.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_score_distributions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Score Distribution: Fraud vs Legitimate",
    save_path: Optional[str] = None,
):
    """Overlapping score histograms split by label.

    Legitimate transactions fill the left tail; fraud transactions concentrate
    towards 1.0.  The separation between the two humps is the model's
    discriminating power at a glance.

    Args:
        y_true:    Ground truth binary labels (0=legit, 1=fraud).
        y_prob:    Predicted fraud probabilities.
        title:     Plot title.
        save_path: If provided, save figure to this path (PNG/PDF/SVG).

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))

    legit_scores = y_prob[y_true == 0]
    fraud_scores = y_prob[y_true == 1]

    bins = np.linspace(0, 1, 60)
    ax.hist(legit_scores, bins=bins, density=True, alpha=0.55,
            color="#4C9BE8", label=f"Legitimate (n={len(legit_scores):,})")
    ax.hist(fraud_scores, bins=bins, density=True, alpha=0.70,
            color="#E84C4C", label=f"Fraud (n={len(fraud_scores):,})")

    ax.set_xlabel("Predicted Fraud Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper center")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Score distribution plot saved to %s", save_path)

    return fig


def plot_dollar_recall_curve(
    sweep_rows: List[dict],
    title: str = "Dollar Recall & Count Recall vs FPR",
    save_path: Optional[str] = None,
):
    """Dollar recall and count recall as a function of FPR operating point.

    Takes the output of ``fpr_sweep()`` directly.  When dollar_recall is
    present, both curves are drawn; otherwise only count recall is shown.
    This lets reviewers directly answer: "if we allow 2% FPR, what fraction
    of fraud dollars do we catch?"

    Args:
        sweep_rows: List of dicts returned by ``evaluation.metrics.fpr_sweep``.
        title:      Plot title.
        save_path:  If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fpr_pcts     = [r["actual_fpr_pct"] for r in sweep_rows]
    recall       = [r["recall"]         for r in sweep_rows]
    has_dollars  = "dollar_recall" in sweep_rows[0]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(fpr_pcts, recall, marker="o", linewidth=2,
            color="#4C9BE8", label="Count Recall (TPR)")

    if has_dollars:
        dollar_recall = [r["dollar_recall"] for r in sweep_rows]
        ax.plot(fpr_pcts, dollar_recall, marker="s", linewidth=2,
                linestyle="--", color="#E84C4C", label="Dollar Recall")

    ax.set_xlabel("Actual FPR (%)")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Dollar recall curve saved to %s", save_path)

    return fig


def plot_model_comparison(
    model_results: dict,
    fpr_target: float = 0.02,
    title: str = "Model Comparison at Fixed FPR",
    save_path: Optional[str] = None,
):
    """Bar chart comparing recall and dollar recall across models at a fixed FPR.

    Args:
        model_results: Dict mapping model name → list of fpr_sweep rows.
                       e.g. {"xgboost": [...], "mlp_xgboost": [...], ...}
        fpr_target:    The FPR operating point to compare (fraction, e.g. 0.02).
        title:         Plot title.
        save_path:     If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    target_pct = round(fpr_target * 100, 2)
    names, recalls, dollar_recalls = [], [], []

    for model_name, rows in model_results.items():
        # Find the row closest to the target FPR.
        row = min(rows, key=lambda r: abs(r["actual_fpr_pct"] - target_pct))
        names.append(model_name)
        recalls.append(row["recall"])
        has_dollars = "dollar_recall" in row
        if has_dollars:
            dollar_recalls.append(row["dollar_recall"])

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2 * (1 + bool(dollar_recalls)), recalls,
           width, label="Count Recall", color="#4C9BE8")
    if dollar_recalls:
        ax.bar(x + width / 2, dollar_recalls,
               width, label="Dollar Recall", color="#E84C4C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{title}  (FPR ≈ {target_pct}%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison plot saved to %s", save_path)

    return fig
