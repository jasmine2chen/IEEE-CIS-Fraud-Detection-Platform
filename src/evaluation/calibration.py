"""Probability calibration utilities for fraud detection models.

Post-processing methods that bring raw model scores into alignment with
empirical fraud rates — a prerequisite for threshold setting and expected-
value decision frameworks.

Provides
--------
calibrate_platt       Logistic regression (Platt scaling) on held-out scores.
calibrate_isotonic    Isotonic regression — non-parametric, more flexible.
bootstrap_ci          Bootstrap confidence intervals for any scalar metric.
fairness_parity_check Demographic parity check across proxy group columns.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def calibrate_platt(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[LogisticRegression, np.ndarray]:
    """Fit a Platt scaling (logistic regression) calibrator on scored data.

    Platt scaling fits a logistic regression on the raw model scores to
    produce calibrated probabilities.  Works best when miscalibration is
    approximately monotone and sigmoid-shaped.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Raw model output probabilities (uncalibrated).

    Returns:
        (calibrator, calibrated_probs): fitted LogisticRegression + calibrated array.
    """
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(y_prob.reshape(-1, 1), y_true)
    calibrated = cal.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    raw_brier   = brier_score_loss(y_true, y_prob)
    cal_brier   = brier_score_loss(y_true, calibrated)
    logger.info(
        "Platt scaling — Brier before: %.4f → after: %.4f (Δ %.4f)",
        raw_brier, cal_brier, cal_brier - raw_brier,
    )
    return cal, calibrated


def apply_platt(
    calibrator: LogisticRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply a fitted Platt calibrator to new scores."""
    return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------

def calibrate_isotonic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[IsotonicRegression, np.ndarray]:
    """Fit an isotonic regression calibrator.

    Non-parametric and more flexible than Platt scaling — can correct
    non-monotone miscalibration.  Requires a larger calibration set to
    avoid overfitting (≥500 samples recommended).

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Raw model output probabilities.

    Returns:
        (calibrator, calibrated_probs): fitted IsotonicRegression + calibrated array.
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(y_prob, y_true)
    calibrated = cal.predict(y_prob)

    raw_brier = brier_score_loss(y_true, y_prob)
    cal_brier = brier_score_loss(y_true, calibrated)
    logger.info(
        "Isotonic calibration — Brier before: %.4f → after: %.4f (Δ %.4f)",
        raw_brier, cal_brier, cal_brier - raw_brier,
    )
    return cal, calibrated


def apply_isotonic(
    calibrator: IsotonicRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply a fitted isotonic calibrator to new scores."""
    return calibrator.predict(y_prob)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for any scalar metric.

    Resamples (y_true, y_prob) pairs with replacement and computes the
    metric on each resample.  Returns the point estimate and the
    percentile-method CI.

    Args:
        y_true:       Ground-truth labels.
        y_prob:       Model scores.
        metric_fn:    Callable(y_true, y_prob) → float. Examples:
                        roc_auc_score, brier_score_loss,
                        lambda yt, yp: fpr_sweep(yt, yp)[3]["recall"]
        n_bootstraps: Number of resample iterations (default 1000).
        ci_level:     Confidence level, e.g. 0.95 for 95% CI.
        seed:         RNG seed for reproducibility.

    Returns:
        dict with keys: point_estimate, ci_lower, ci_upper, std.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)

    scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass  # skip degenerate resamples (e.g. all one class for AUC)

    arr = np.array(scores)
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point_estimate": float(point),
        "ci_lower":       float(np.percentile(arr, 100 * alpha)),
        "ci_upper":       float(np.percentile(arr, 100 * (1 - alpha))),
        "std":            float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Fairness parity check
# ---------------------------------------------------------------------------

def fairness_parity_check(
    df: pd.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_cols: List[str],
    threshold: float = 0.5,
    min_group_size: int = 50,
) -> pd.DataFrame:
    """Demographic parity and equalized-odds check across proxy group columns.

    Computes per-group fraud rate, predicted positive rate, and FPR at a
    fixed threshold.  Groups with fewer than min_group_size samples are
    excluded as statistically unreliable.

    This is a diagnostic tool — not a decision tool.  Disparate impact in
    proxy variables (e.g. billing region, email domain) should be reviewed
    with a domain expert before acting.

    Args:
        df:             DataFrame containing scores and labels.
        y_prob_col:     Column with model fraud probabilities.
        y_true_col:     Column with binary fraud labels.
        group_cols:     List of columns to group by (e.g. ["addr2", "ProductCD"]).
        threshold:      Decision threshold for predicted positive rate / FPR.
        min_group_size: Exclude groups smaller than this.

    Returns:
        DataFrame with columns: group_col, group_value, n, fraud_rate,
        predicted_positive_rate, fpr, tpr, parity_ratio
        (predicted_positive_rate / overall_predicted_positive_rate).
    """
    y_prob = df[y_prob_col].values
    y_true = df[y_true_col].values
    y_pred = (y_prob >= threshold).astype(int)

    overall_ppr = float(y_pred.mean())
    rows = []

    for col in group_cols:
        if col not in df.columns:
            logger.warning("Fairness check: column '%s' not found — skipping.", col)
            continue

        for val, grp_idx in df.groupby(col).groups.items():
            grp_idx_arr = np.array(grp_idx)
            n = len(grp_idx_arr)
            if n < min_group_size:
                continue

            yt = y_true[grp_idx_arr]
            yp = y_pred[grp_idx_arr]

            fraud_rate = float(yt.mean())
            ppr        = float(yp.mean())
            neg_mask   = yt == 0
            pos_mask   = yt == 1

            fpr = float(yp[neg_mask].mean()) if neg_mask.sum() > 0 else float("nan")
            tpr = float(yp[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan")
            parity_ratio = ppr / overall_ppr if overall_ppr > 0 else float("nan")

            rows.append({
                "group_col":              col,
                "group_value":            str(val),
                "n":                      n,
                "fraud_rate":             round(fraud_rate, 4),
                "predicted_positive_rate": round(ppr, 4),
                "fpr":                    round(fpr, 4),
                "tpr":                    round(tpr, 4),
                "parity_ratio":           round(parity_ratio, 4),
            })

    result_df = pd.DataFrame(rows).sort_values(
        ["group_col", "parity_ratio"], ascending=[True, False]
    ).reset_index(drop=True)

    if result_df.empty:
        logger.warning("Fairness check returned no rows — check group_cols and min_group_size.")
    else:
        flagged = result_df[result_df["parity_ratio"] < 0.8]
        if not flagged.empty:
            logger.warning(
                "Fairness check: %d group(s) have parity_ratio < 0.8 "
                "(predicted positive rate < 80%% of overall). Review before deployment:\n%s",
                len(flagged),
                flagged[["group_col", "group_value", "n", "parity_ratio"]].to_string(index=False),
            )

    return result_df
