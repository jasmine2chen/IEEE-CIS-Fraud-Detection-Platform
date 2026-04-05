from typing import Any, Dict, Optional, Tuple

import numpy as np
from xgboost import XGBClassifier


def make_fpr_eval_metric(threshold: float = 0.85):
    """Return a custom XGBoost eval metric that computes FPR at a fixed threshold.

    Using FPR as the early stopping signal instead of AUC aligns training
    directly with the business constraint (FP rate < 15%) and stops earlier:
    AUC shifts gradually with every tree; FPR at a high threshold (0.85)
    plateaus quickly once the model learns to be conservative, triggering
    early stopping in fewer rounds.

    XGBoost convention: lower return value = better. FPR is naturally
    minimised, so no sign flip is needed.

    The function receives raw logit scores (before sigmoid) because
    XGBClassifier uses binary:logistic internally.
    """
    def fpr_at_threshold(predt: np.ndarray, dtrain) -> Tuple[str, float]:
        labels = dtrain.get_label()
        # predt is raw logit — apply sigmoid to get probabilities
        probs = 1.0 / (1.0 + np.exp(-predt))
        preds_binary = (probs >= threshold).astype(int)

        negatives = labels == 0
        fp = int(((preds_binary == 1) & negatives).sum())
        tn = int(((preds_binary == 0) & negatives).sum())
        fpr = fp / (fp + tn + 1e-8)  # epsilon guards against all-fraud eval sets

        return "fpr_at_threshold", float(fpr)

    return fpr_at_threshold


def get_xgboost_model(
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    fpr_threshold: Optional[float] = None,
) -> XGBClassifier:
    """Instantiate an XGBoost classifier with business-aligned early stopping.

    Args:
        params: Dict of XGBoost hyperparameters. When called from train.py
                this is populated from configs/model_config.yaml so there is
                a single source of truth for all hyperparameters.
                Falls back to conservative defaults when called standalone.
        early_stopping_rounds: Stop training when the eval metric does not
                improve for this many consecutive rounds. n_estimators becomes
                an upper bound, not a fixed count.
        fpr_threshold: When provided, early stopping monitors FPR at this
                operating threshold instead of AUC. Set to
                serving.fraud_threshold_prob from config to align training
                directly with the production decision boundary.
    """
    if params is None:
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 9,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        }

    params = dict(params)  # don't mutate caller's dict

    if fpr_threshold is not None:
        # Replace generic AUC with the business-aligned FPR metric.
        # XGBoost uses the last entry in eval_metric for early stopping.
        params.pop("eval_metric", None)
        eval_metric = make_fpr_eval_metric(fpr_threshold)
    else:
        eval_metric = params.pop("eval_metric", "auc")

    return XGBClassifier(
        **params,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
    )
