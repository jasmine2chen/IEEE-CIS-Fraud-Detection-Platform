"""Feature group ablation study — measure marginal contribution of each feature group.

Systematically disables individual feature groups while holding the model architecture
constant (XGBoost throughout), isolating the contribution of each group to the primary
business metric (recall @ 2% FPR).

This is the artifact a research engineer uses to answer:
  "Which features are actually earning their keep? What would we lose if we simplified?"

Design decisions
----------------
- XGBoost is always the classifier (controls for model architecture).  Using a hybrid
  model would confound feature contributions with encoder design choices.
- Each variant is trained from scratch on the same train/OOT split as the baseline.
- Ablation is applied by zeroing out (not dropping) columns: this preserves the fitted
  pipeline and avoids shape mismatches in the ColumnTransformer.
- Each variant is logged as a child MLflow run under "fraud_ablation" experiment so
  results are queryable alongside training runs.

Usage
-----
    python -m src.evaluation.ablation \\
        --trans  data/raw/train_transaction.csv \\
        --id     data/raw/train_identity.csv \\
        --output reports/ablation

    make ablation
"""

import argparse
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import mlflow
import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocessing.data_loader import prepare_data
from src.training.train import time_consistency_split
from src.evaluation.metrics import evaluate_classification, fpr_sweep
from src.feature_engineering.build_features import get_full_pipeline
from src.training.models.tree_models import get_xgboost_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """Specification for one ablation experiment.

    Args:
        name:          Short identifier used in output filenames and MLflow tags.
        description:   Human-readable description of what was removed.
        zero_out_prefixes: Feature name prefixes to zero out after pipeline.transform().
            Zeroing (not dropping) preserves the fitted pipeline's output shape.
        rationale:     Why this group is worth testing.
    """
    name: str
    description: str
    zero_out_prefixes: List[str] = field(default_factory=list)
    rationale: str = ""


# The canonical ablation suite.  Each variant removes one interpretable feature group.
ABLATION_SUITE: List[AblationVariant] = [
    AblationVariant(
        name="baseline",
        description="Full pipeline — all feature groups enabled",
        zero_out_prefixes=[],
        rationale="Reference point. All other deltas computed relative to this.",
    ),
    AblationVariant(
        name="no_uid_velocity",
        description="Drop velocity features (1h / 24h transaction counts per UID)",
        zero_out_prefixes=["uid_txn_count_1h", "uid_txn_count_24h"],
        rationale="Velocity caps are expensive to compute in real-time. "
                  "Quantifies whether the latency cost is justified.",
    ),
    AblationVariant(
        name="no_uid_aggregations",
        description="Drop all UID aggregation features (velocity + amount stats + consistency)",
        zero_out_prefixes=[
            "uid_txn_count", "uid_days_since_first", "uid_time_since",
            "uid_avg_time", "uid_unique", "uid_is_email",
            "TransactionAmt_uid", "M1_uid", "M4_uid", "M5_uid",
            "M7_uid", "M8_uid", "M9_uid", "C1_uid", "C9_uid",
            "C11_uid", "C13_uid", "D2_uid", "D4_uid", "D9_uid",
            "D10_uid", "D15_uid", "uid_transaction_count",
        ],
        rationale="UID aggregations require the full transaction history at serve time. "
                  "Batch-only systems may not have this. Measures the UID feature group's total lift.",
    ),
    AblationVariant(
        name="no_deviation_features",
        description="Drop amount deviation features (deviation from UID mean, percentile rank, outlier flag)",
        zero_out_prefixes=[
            "TransactionAmt_deviation_from_uid_mean",
            "TransactionAmt_ratio_to_uid_mean",
            "TransactionAmt_is_uid_outlier",
            "TransactionAmt_percentile_in_uid",
        ],
        rationale="Deviation features are second-order (require uid_mean to be computed first). "
                  "Tests whether the extra complexity over raw UID amount stats is worth it.",
    ),
    AblationVariant(
        name="no_d_normalization",
        description="Drop normalized D-column features (temporal delta normalization)",
        zero_out_prefixes=["_normalized"],
        rationale="D-column normalization removes calendar drift. "
                  "Measures how much the drift correction matters for OOT generalization.",
    ),
    AblationVariant(
        name="no_c_columns",
        description="Drop C-column UID aggregation features",
        zero_out_prefixes=["C1_uid", "C9_uid", "C11_uid", "C13_uid"],
        rationale="C-columns are count aggregates of various transaction flags. "
                  "Isolates their contribution from the amount-based UID stats.",
    ),
    AblationVariant(
        name="raw_features_only",
        description="Drop all engineered features — raw transaction fields only",
        zero_out_prefixes=[
            "uid_", "TransactionAmt_uid", "TransactionAmt_deviation",
            "TransactionAmt_ratio", "TransactionAmt_is_uid", "TransactionAmt_percentile",
            "_normalized", "M1_uid", "M4_uid", "M5_uid", "M7_uid",
            "M8_uid", "M9_uid", "C1_uid", "C9_uid", "C11_uid", "C13_uid",
        ],
        rationale="Lower bound: a model trained only on raw IEEE-CIS fields, "
                  "with no feature engineering. Shows the total lift from the feature pipeline.",
    ),
]


# ---------------------------------------------------------------------------
# Core ablation logic
# ---------------------------------------------------------------------------

def _zero_out_features(
    X_proc: np.ndarray,
    feature_names: List[str],
    prefixes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Zero out columns whose names match any of the given prefixes.

    Returns (zeroed_array, list_of_zeroed_feature_names).
    """
    if not prefixes:
        return X_proc.copy(), []

    zeroed = X_proc.copy()
    zeroed_names: List[str] = []
    for i, name in enumerate(feature_names):
        if any(name.startswith(p) or p in name for p in prefixes):
            zeroed[:, i] = 0.0
            zeroed_names.append(name)

    return zeroed, zeroed_names


def _get_feature_names(pipeline: Any, X_sample: pd.DataFrame) -> List[str]:
    """Extract feature names from the fitted sklearn Pipeline.

    Falls back to positional names if get_feature_names_out is unavailable.
    """
    try:
        prep = pipeline.named_steps["preprocessing"]
        return list(prep.get_feature_names_out())
    except Exception:
        n_features = pipeline.transform(X_sample.iloc[:1]).shape[1]
        return [f"f{i}" for i in range(n_features)]


def run_ablation_variant(
    variant: AblationVariant,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    xgb_params: Dict[str, Any],
    fpr_threshold: float,
    amounts: Optional[np.ndarray],
    tracking_uri: str,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Train XGBoost on the ablated feature matrix and evaluate.

    Each variant is logged as a child MLflow run so results are queryable.

    Returns a result dict with variant metadata + evaluation metrics.
    """
    logger.info("Running ablation: %s — %s", variant.name, variant.description)

    X_train_abl, zeroed_names = _zero_out_features(
        X_train_proc, feature_names, variant.zero_out_prefixes
    )
    X_test_abl, _ = _zero_out_features(
        X_test_proc, feature_names, variant.zero_out_prefixes
    )
    n_zeroed = len(zeroed_names)
    logger.info("  Zeroed %d feature columns.", n_zeroed)

    model = get_xgboost_model(
        params=xgb_params,
        early_stopping_rounds=50,
        fpr_threshold=fpr_threshold,
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment("fraud_ablation")
        with mlflow.start_run(run_name=f"ablation_{variant.name}"):
            mlflow.set_tag("ablation_variant", variant.name)
            mlflow.set_tag("ablation_description", variant.description)
            mlflow.log_param("n_zeroed_features", n_zeroed)
            if zeroed_names:
                mlflow.log_param("zeroed_feature_sample", str(zeroed_names[:10]))

            model.fit(
                X_train_abl, y_train,
                eval_set=[(X_test_abl, y_test)],
                verbose=False,
            )

            y_arr = y_test if isinstance(y_test, np.ndarray) else y_test.values
            probs = model.predict_proba(X_test_abl)[:, 1]
            metrics = evaluate_classification(y_arr, probs, threshold=0.5)
            sweep = fpr_sweep(y_arr, probs, amounts=amounts, fpr_targets=[0.01, 0.02, 0.05])
            recall_by_fpr = {row["target_fpr_pct"]: row for row in sweep}

            r2  = recall_by_fpr.get(2.0, {}).get("recall", 0.0)
            dr2 = recall_by_fpr.get(2.0, {}).get("dollar_recall", 0.0)

            mlflow.log_metrics({
                "roc_auc":              metrics["roc_auc"],
                "pr_auc":               metrics["pr_auc"],
                "auc_at_5pct_fpr":      metrics["auc_at_max_fpr"],
                "recall_at_2pct_fpr":   r2,
                "dollar_recall_2pct":   dr2,
                "brier_score":          metrics["brier_score"],
            })

    except Exception as exc:
        logger.warning("MLflow logging failed for variant '%s': %s", variant.name, exc)
        y_arr = y_test if isinstance(y_test, np.ndarray) else y_test.values
        probs = model.predict_proba(X_test_abl)[:, 1]
        metrics = evaluate_classification(y_arr, probs, threshold=0.5)
        sweep = fpr_sweep(y_arr, probs, amounts=amounts, fpr_targets=[0.01, 0.02, 0.05])
        recall_by_fpr = {row["target_fpr_pct"]: row for row in sweep}
        r2  = recall_by_fpr.get(2.0, {}).get("recall", 0.0)
        dr2 = recall_by_fpr.get(2.0, {}).get("dollar_recall", 0.0)

    result = {
        "variant_name":        variant.name,
        "description":         variant.description,
        "n_zeroed_features":   n_zeroed,
        "zeroed_feature_sample": zeroed_names[:10],
        "roc_auc":             round(metrics["roc_auc"], 4),
        "pr_auc":              round(metrics["pr_auc"], 4),
        "auc_at_5pct_fpr":     round(metrics["auc_at_max_fpr"], 4),
        "recall_at_2pct_fpr":  round(r2, 4),
        "dollar_recall_2pct":  round(dr2, 4),
        "brier_score":         round(metrics["brier_score"], 4),
    }

    if baseline_metrics:
        result["delta_recall_2pct"]   = round(r2  - baseline_metrics.get("recall_at_2pct_fpr", 0), 4)
        result["delta_roc_auc"]       = round(metrics["roc_auc"] - baseline_metrics.get("roc_auc", 0), 4)

    logger.info(
        "  %s — AUC: %.4f  Recall@2%%FPR: %.4f  delta_recall: %s",
        variant.name, metrics["roc_auc"], r2,
        f"{result.get('delta_recall_2pct', 'N/A'):+.4f}" if "delta_recall_2pct" in result else "N/A",
    )
    return result


def run_ablation_suite(
    variants: Optional[List[AblationVariant]],
    trans_path: str,
    id_path: str,
    output_dir: str = "reports/ablation",
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run all ablation variants and write results to disk."""
    if variants is None:
        variants = ABLATION_SUITE

    cfg = load_config(config_path)
    tracking_uri = cfg["training"].get("mlflow_tracking_uri", "")
    xgb_params = cfg.get("xgboost_params", {})
    fpr_threshold = cfg["serving"]["fraud_threshold_prob"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data for ablation study...")
    X, y = prepare_data(trans_path, id_path)
    train_idx, test_idx = time_consistency_split(X)
    X_train_raw, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test_raw,  y_test  = X.loc[test_idx],  y.loc[test_idx]
    amounts = (
        X_test_raw["TransactionAmt"].values
        if "TransactionAmt" in X_test_raw.columns else None
    )
    y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
    y_test_arr  = y_test.values  if hasattr(y_test,  "values") else y_test

    logger.info("Fitting feature pipeline once (shared across all variants)...")
    pipeline = get_full_pipeline()
    X_train_proc = pipeline.fit_transform(X_train_raw)
    X_test_proc  = pipeline.transform(X_test_raw)
    feature_names = _get_feature_names(pipeline, X_train_raw)
    logger.info("Feature matrix: %d train, %d test, %d features",
                X_train_proc.shape[0], X_test_proc.shape[0], X_train_proc.shape[1])

    results = []
    baseline_metrics: Optional[Dict[str, float]] = None

    for variant in variants:
        result = run_ablation_variant(
            variant=variant,
            X_train_proc=X_train_proc,
            y_train=y_train_arr,
            X_test_proc=X_test_proc,
            y_test=y_test_arr,
            feature_names=feature_names,
            xgb_params=xgb_params,
            fpr_threshold=fpr_threshold,
            amounts=amounts,
            tracking_uri=tracking_uri,
            baseline_metrics=baseline_metrics,
        )
        results.append(result)

        if variant.name == "baseline":
            baseline_metrics = {
                "recall_at_2pct_fpr": result["recall_at_2pct_fpr"],
                "roc_auc":            result["roc_auc"],
            }

    df = pd.DataFrame(results)

    csv_path = os.path.join(output_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Ablation results written to %s", csv_path)

    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("ABLATION RESULTS — delta_recall_2pct = change in Recall@2%FPR vs full pipeline")
    print("=" * 90)
    display_cols = ["variant_name", "roc_auc", "recall_at_2pct_fpr",
                    "dollar_recall_2pct", "delta_recall_2pct", "n_zeroed_features"]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Feature group ablation study for fraud detection models"
    )
    parser.add_argument("--trans",  required=True, help="Path to raw transaction CSV")
    parser.add_argument("--id",     required=True, help="Path to raw identity CSV")
    parser.add_argument(
        "--output",
        default="reports/ablation",
        help="Output directory (default: reports/ablation)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variant names to run (default: full ABLATION_SUITE)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/model_config.yaml)",
    )
    args = parser.parse_args()

    selected: Optional[List[AblationVariant]] = None
    if args.variants:
        name_set = set(args.variants)
        selected = [v for v in ABLATION_SUITE if v.name in name_set]
        if not selected:
            raise ValueError(
                f"None of {args.variants} matched known variants: "
                f"{[v.name for v in ABLATION_SUITE]}"
            )

    run_ablation_suite(
        variants=selected,
        trans_path=args.trans,
        id_path=args.id,
        output_dir=args.output,
        config_path=args.config,
    )
