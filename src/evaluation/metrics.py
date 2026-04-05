import logging
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss, roc_curve, auc,
)

logger = logging.getLogger(__name__)


def auc_at_max_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.05) -> float:
    """Partial AUC up to max_fpr, normalized so random = 0.5 and perfect = 1.0.

    Standard ROC-AUC averages performance across all FPR levels, but in fraud
    detection the operating regime is 0–5% FPR (anything above that is
    operationally unacceptable). Normalizing by max_fpr maps the metric to
    [0, 1]: a random classifier scores 0.5, a perfect classifier scores 1.0.

    Args:
        y_true:  Ground truth binary labels.
        y_prob:  Predicted probabilities for the positive class.
        max_fpr: Upper bound on FPR to include in the partial AUC. Default 5%.

    Returns:
        Normalized partial AUC in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Clip curve at max_fpr and interpolate the endpoint so the integral is exact.
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    fpr_clip = np.append(fpr[:stop], max_fpr)
    tpr_clip = np.append(tpr[:stop], float(np.interp(max_fpr, fpr, tpr)))
    return float(auc(fpr_clip, tpr_clip) / max_fpr)


def fpr_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    fpr_targets: Optional[List[float]] = None,
) -> List[dict]:
    """Evaluate model performance across a range of operating FPR thresholds.

    Mimics the ``score_analysis`` function used in production fraud systems:
    sweeps from very tight (0.1% FPR) to relaxed (25% FPR), returning the
    recall, precision, and optionally dollar recall at each operating point.
    This replaces the arbitrary 0.5 threshold with business-relevant cut-offs.

    Args:
        y_true:      Ground truth binary labels.
        y_prob:      Predicted probabilities for the positive class.
        amounts:     Transaction amounts aligned with y_true. When provided,
                     each row includes dollar_recall (fraction of total fraud
                     dollars caught) in addition to count-based recall.
        fpr_targets: FPR levels to evaluate. Defaults to standard sweep from
                     0.1% to 25%.

    Returns:
        List of dicts, one per FPR target, with keys:
        target_fpr_pct, threshold, actual_fpr_pct, recall, precision,
        tp, fp, fn, tn, [dollar_recall, caught_fraud_dollars, total_fraud_dollars].
    """
    if fpr_targets is None:
        fpr_targets = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    # fpr_arr[0]=0, tpr_arr[0]=0 is the sentinel (classify-nothing) point.
    # thresholds[i] <-> fpr_arr[i+1], tpr_arr[i+1]  (len(thresholds) = len(fpr_arr) - 1).

    rows = []
    for target_fpr in fpr_targets:
        # Largest ROC index where actual FPR is still <= target (stay within budget).
        candidates = np.where(fpr_arr <= target_fpr)[0]
        idx = int(candidates[-1]) if len(candidates) else 0

        actual_fpr = float(fpr_arr[idx])
        actual_tpr = float(tpr_arr[idx])

        # Map ROC index to the decision threshold (offset by 1 due to sentinel).
        thresh_idx = max(0, min(idx - 1, len(thresholds) - 1))
        thresh = float(thresholds[thresh_idx]) if idx > 0 else float(thresholds[0])

        y_pred = (y_prob >= thresh).astype(int)
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        row: dict = {
            "target_fpr_pct": round(target_fpr * 100, 2),
            "threshold":      round(thresh, 4),
            "actual_fpr_pct": round(actual_fpr * 100, 2),
            "recall":         round(actual_tpr, 4),
            "precision":      round(precision, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

        if amounts is not None:
            fraud_mask        = y_true == 1
            total_fraud_usd   = float(amounts[fraud_mask].sum())
            caught_fraud_usd  = float(amounts[(y_pred == 1) & fraud_mask].sum())
            row["dollar_recall"]        = round(caught_fraud_usd / total_fraud_usd, 4) if total_fraud_usd else 0.0
            row["caught_fraud_dollars"] = round(caught_fraud_usd, 2)
            row["total_fraud_dollars"]  = round(total_fraud_usd, 2)

        rows.append(row)

    return rows


def log_fpr_sweep(rows: List[dict]) -> None:
    """Emit a formatted FPR sweep table via the logging framework."""
    has_dollars = "dollar_recall" in rows[0]
    header = (
        f"{'FPR%':>6}  {'Thresh':>7}  {'Act.FPR%':>8}  "
        f"{'Recall':>6}  {'Precision':>9}  {'TP':>6}  {'FP':>6}"
    )
    if has_dollars:
        header += f"  {'$Recall':>7}"

    lines = ["\n=== FPR Sweep ===", header, "-" * len(header)]
    for r in rows:
        line = (
            f"{r['target_fpr_pct']:>6.2f}  {r['threshold']:>7.4f}  "
            f"{r['actual_fpr_pct']:>8.2f}  {r['recall']:>6.4f}  "
            f"{r['precision']:>9.4f}  {r['tp']:>6d}  {r['fp']:>6d}"
        )
        if has_dollars:
            line += f"  {r['dollar_recall']:>7.4f}"
        lines.append(line)

    lines.append("=================")
    logger.info("\n".join(lines))


def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    max_fpr: float = 0.05,
) -> dict:
    """Compute a comprehensive set of binary classification metrics.

    Includes standard metrics plus fraud-specific additions:
    - auc_at_max_fpr: partial AUC up to max_fpr (default 5%), normalized to [0,1].
      More informative than global AUC when the operating regime is low-FPR only.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "precision":      float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":       float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":        float(roc_auc_score(y_true, y_prob)),
        "pr_auc":         float(average_precision_score(y_true, y_prob)),
        "auc_at_max_fpr": auc_at_max_fpr(y_true, y_prob, max_fpr=max_fpr),
        "brier_score":    float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def log_evaluation_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> None:
    """Emit a structured evaluation report via the logging framework."""
    y_pred = (y_prob >= threshold).astype(int)
    report  = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])
    metrics = evaluate_classification(y_true, y_prob, threshold)
    cm      = metrics["confusion_matrix"]
    logger.info(
        "\n=== Evaluation Report ===\n%s\n"
        "ROC-AUC:        %.4f\n"
        "PR-AUC:         %.4f\n"
        "AUC@5%%FPR:     %.4f\n"
        "Brier Score:    %.4f\n"
        "Confusion Matrix:\n"
        "  TN: %d | FP: %d\n"
        "  FN: %d | TP: %d\n"
        "=========================",
        report,
        metrics["roc_auc"], metrics["pr_auc"],
        metrics["auc_at_max_fpr"], metrics["brier_score"],
        cm[0][0], cm[0][1], cm[1][0], cm[1][1],
    )
