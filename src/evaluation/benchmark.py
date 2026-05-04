"""Model benchmarking harness — compare champion models on an identical test set.

Loads the @champion artifact for each requested model type from the MLflow Model
Registry, evaluates on the same out-of-time split used during training, and
produces a side-by-side comparison table covering discrimination, business-aligned
operating point performance, calibration, and inference cost.

Usage
-----
    python -m src.evaluation.benchmark \\
        --trans  data/raw/train_transaction.csv \\
        --id     data/raw/train_identity.csv \\
        --models xgboost mlp_xgboost \\
        --output reports/benchmark

    make benchmark
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocessing.data_loader import prepare_data
from src.evaluation.metrics import (
    evaluate_classification,
    fpr_sweep,
    auc_at_max_fpr,
)
from src.feature_engineering.build_features import get_full_pipeline
from src.deployment import registry

logger = logging.getLogger(__name__)

VALID_MODELS = ("xgboost", "mlp_xgboost")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Evaluation results for one model type on the OOT test set."""
    model_name: str

    # --- Discrimination ---
    roc_auc: float
    pr_auc: float
    auc_at_5pct_fpr: float          # partial AUC within operational range

    # --- Business-aligned recall at operating FPR points ---
    recall_at_1pct_fpr: float
    recall_at_2pct_fpr: float
    recall_at_5pct_fpr: float
    dollar_recall_at_2pct_fpr: float  # fraction of fraud *dollars* caught

    # --- Calibration ---
    brier_score: float

    # --- Production cost ---
    inference_latency_p50_ms: float   # median single-sample latency
    inference_latency_p99_ms: float   # p99 single-sample latency
    n_estimators: int                  # proxy for model complexity

    # --- Traceability ---
    run_id: str
    model_version: str
    artifact_source: str              # "registry" or "disk"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_artifacts(
    model_type: str,
    tracking_uri: str,
) -> Tuple[Any, Any, str, str]:
    """Load feature pipeline + XGBoost model from MLflow registry or disk.

    Returns (pipeline, model, run_id, artifact_source).
    """
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            pipeline, model = registry.load_champion(model_type, tracking_uri=tracking_uri)
            run_id = registry.get_champion_run_id(model_type, tracking_uri=tracking_uri) or "unknown"
            logger.info("Loaded @champion '%s' from MLflow registry.", model_type)
            return pipeline, model, run_id, "registry"
        except Exception as exc:
            logger.warning("Registry load failed for '%s' (%s) — falling back to disk.", model_type, exc)

    _DISK_MODEL_PATHS = {
        "xgboost":    "models/xgboost_fraud_model.joblib",
        "mlp_xgboost": "models/mlp_xgboost/xgboost.joblib",
    }
    model_path = _DISK_MODEL_PATHS.get(model_type, "models/xgboost_fraud_model.joblib")
    pipeline = joblib.load("models/feature_pipeline.joblib")
    model = joblib.load(model_path)
    logger.info("Loaded '%s' artifacts from disk (%s).", model_type, model_path)
    return pipeline, model, "disk", "disk"


def _load_encoder(model_type: str) -> Optional[Any]:
    """Load the MLP encoder for mlp_xgboost from disk. Returns None for xgboost."""
    if model_type != "mlp_xgboost":
        return None

    encoder_path = "models/mlp_xgboost/encoder.pt"
    if not Path(encoder_path).exists():
        return None

    import torch
    from src.training.models.mlp_tree import MLPEncoder
    ckpt = torch.load(encoder_path, map_location="cpu")
    enc = MLPEncoder(input_dim=ckpt["input_dim"], hidden_dims=(256, 128, 64))
    enc.load_state_dict(ckpt["model_state_dict"])
    enc.eval()
    return enc


def _build_input(
    model_type: str,
    encoder: Optional[Any],
    X_proc: np.ndarray,
    X_raw: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Return the feature matrix the XGBoost stage expects.

    For xgboost: just the pipeline-processed features.
    For mlp_xgboost: [pipeline features || encoder embeddings].
    """
    if encoder is None or model_type == "xgboost":
        return X_proc

    from src.training.models.mlp_tree import extract_mlp_embeddings
    embeddings = extract_mlp_embeddings(encoder, X_proc, device="cpu")
    return np.hstack([X_proc, embeddings])


def _get_model_version(model_type: str, tracking_uri: str) -> str:
    if not tracking_uri:
        return "disk"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        model_name = registry.get_model_name(model_type)
        mv = client.get_model_version_by_alias(name=model_name, alias="champion")
        return str(mv.version)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# OOT split (mirrors src/training/train.py exactly)
# ---------------------------------------------------------------------------

def _get_oot_split(
    trans_path: str,
    id_path: str,
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """Reproduce the same OOT test set used during training."""
    from src.training.train import time_consistency_split

    X, y = prepare_data(trans_path, id_path)
    _, test_idx = time_consistency_split(X)
    X_test_raw = X.loc[test_idx]
    y_test = y.loc[test_idx]
    amounts = (
        X_test_raw["TransactionAmt"].values
        if "TransactionAmt" in X_test_raw.columns else None
    )
    return X_test_raw, y_test, amounts


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def _measure_latency(
    model_type: str,
    pipeline: Any,
    encoder: Optional[Any],
    model: Any,
    X_sample: pd.DataFrame,
    n_reps: int = 200,
) -> Tuple[float, float]:
    """Measure end-to-end single-sample predict_proba latency (ms).

    Includes pipeline transform + optional encoder embedding extraction + XGBoost.
    Runs n_reps timed calls on a single random row.
    Returns (p50_ms, p99_ms). First 10 reps discarded as warm-up.
    """
    row = X_sample.iloc[[0]]
    timings = []
    for i in range(n_reps + 10):
        t0 = time.perf_counter()
        X_proc = pipeline.transform(row)
        X_input = _build_input(model_type, encoder, X_proc, X_raw=row)
        _ = model.predict_proba(X_input)[:, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= 10:
            timings.append(elapsed_ms)
    arr = np.array(timings)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_type: str,
    pipeline: Any,
    encoder: Optional[Any],
    model: Any,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    amounts: Optional[np.ndarray],
    run_id: str,
    model_version: str,
    artifact_source: str,
    n_latency_reps: int = 200,
) -> ModelResult:
    """Evaluate one model on the OOT test set and return a ModelResult."""
    logger.info("Evaluating '%s' on %d test samples...", model_type, len(X_test_raw))

    X_proc = pipeline.transform(X_test_raw)
    X_input = _build_input(model_type, encoder, X_proc, X_raw=X_test_raw)
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    probs = model.predict_proba(X_input)[:, 1]

    # Core metrics
    metrics = evaluate_classification(y_arr, probs, threshold=0.5, max_fpr=0.05)

    # FPR sweep for recall@1%/2%/5% and dollar recall
    sweep = fpr_sweep(y_arr, probs, amounts=amounts,
                      fpr_targets=[0.01, 0.02, 0.05])
    recall_by_fpr = {row["target_fpr_pct"]: row for row in sweep}

    r1  = recall_by_fpr.get(1.0,  {}).get("recall", 0.0)
    r2  = recall_by_fpr.get(2.0,  {}).get("recall", 0.0)
    r5  = recall_by_fpr.get(5.0,  {}).get("recall", 0.0)
    dr2 = recall_by_fpr.get(2.0,  {}).get("dollar_recall", 0.0)

    # Latency
    p50, p99 = _measure_latency(model_type, pipeline, encoder, model, X_test_raw, n_reps=n_latency_reps)

    # Model complexity proxy
    n_est = getattr(model, "n_estimators", 0)
    best_iter = getattr(model, "best_iteration", None)
    n_trees = int(best_iter) if best_iter is not None else int(n_est)

    logger.info(
        "%s — AUC: %.4f  PR-AUC: %.4f  pAUC@5%%: %.4f  "
        "Recall@2%%FPR: %.4f  $Recall@2%%: %.4f  "
        "Latency p50: %.2fms  p99: %.2fms",
        model_type,
        metrics["roc_auc"], metrics["pr_auc"], metrics["auc_at_max_fpr"],
        r2, dr2, p50, p99,
    )

    return ModelResult(
        model_name=model_type,
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        auc_at_5pct_fpr=metrics["auc_at_max_fpr"],
        recall_at_1pct_fpr=r1,
        recall_at_2pct_fpr=r2,
        recall_at_5pct_fpr=r5,
        dollar_recall_at_2pct_fpr=dr2,
        brier_score=metrics["brier_score"],
        inference_latency_p50_ms=p50,
        inference_latency_p99_ms=p99,
        n_estimators=n_trees,
        run_id=run_id,
        model_version=model_version,
        artifact_source=artifact_source,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ("roc_auc",                  "ROC-AUC",         ".4f"),
    ("pr_auc",                   "PR-AUC",           ".4f"),
    ("auc_at_5pct_fpr",          "pAUC@5%FPR",       ".4f"),
    ("recall_at_1pct_fpr",       "Recall@1%FPR",     ".4f"),
    ("recall_at_2pct_fpr",       "Recall@2%FPR",     ".4f"),
    ("recall_at_5pct_fpr",       "Recall@5%FPR",     ".4f"),
    ("dollar_recall_at_2pct_fpr","$Recall@2%FPR",    ".4f"),
    ("brier_score",              "Brier↓",           ".4f"),
    ("inference_latency_p50_ms", "Latency p50(ms)",  ".2f"),
    ("inference_latency_p99_ms", "Latency p99(ms)",  ".2f"),
    ("n_estimators",             "N Trees",          "d"),
]

_LOWER_IS_BETTER = {"brier_score", "inference_latency_p50_ms", "inference_latency_p99_ms"}


def format_results_table(results: List[ModelResult]) -> str:
    """Render a GitHub-flavoured markdown table. Best value per column bolded."""
    headers = ["Model"] + [col[1] for col in _METRIC_COLS]
    rows = []

    for r in results:
        d = asdict(r)
        row = [r.model_name]
        for attr, _, fmt in _METRIC_COLS:
            val = d[attr]
            row.append(val)
        rows.append(row)

    # Determine best per metric column
    best_idx: List[int] = []
    for col_idx, (attr, _, _) in enumerate(_METRIC_COLS):
        vals = [asdict(r)[attr] for r in results]
        lower_better = attr in _LOWER_IS_BETTER
        best = min(vals) if lower_better else max(vals)
        best_idx.append(vals.index(best))

    # Format rows
    formatted_rows = []
    for row_idx, row in enumerate(rows):
        fmt_row = [row[0]]
        for col_idx, (attr, _, fmt) in enumerate(_METRIC_COLS):
            val = row[col_idx + 1]
            cell = f"{val:{fmt}}"
            if row_idx == best_idx[col_idx]:
                cell = f"**{cell}**"
            fmt_row.append(cell)
        formatted_rows.append(fmt_row)

    # Build markdown
    sep = [":---"] + ["---:" for _ in _METRIC_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in formatted_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.append(
        "\n*↓ = lower is better. Best value per column **bolded**. "
        "All models evaluated on the same OOT test split.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    trans_path: str,
    id_path: str,
    model_types: List[str],
    output_dir: str = "reports/benchmark",
    config_path: Optional[str] = None,
    n_latency_reps: int = 200,
) -> List[ModelResult]:
    """Compare champion models on the OOT test set and write reports."""
    cfg = load_config(config_path)
    tracking_uri = cfg["training"].get("mlflow_tracking_uri", "")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading OOT test split from %s + %s", trans_path, id_path)
    X_test_raw, y_test, amounts = _get_oot_split(trans_path, id_path)
    logger.info("OOT test set: %d samples, %.2f%% fraud",
                len(y_test), 100 * y_test.mean())

    results: List[ModelResult] = []
    for model_type in model_types:
        if model_type not in VALID_MODELS:
            logger.warning("Unknown model type '%s' — skipping.", model_type)
            continue
        try:
            pipeline, model, run_id, source = _load_artifacts(model_type, tracking_uri)
            encoder = _load_encoder(model_type)
            version = _get_model_version(model_type, tracking_uri)
            result = evaluate_model(
                model_type=model_type,
                pipeline=pipeline,
                encoder=encoder,
                model=model,
                X_test_raw=X_test_raw,
                y_test=y_test,
                amounts=amounts,
                run_id=run_id,
                model_version=version,
                artifact_source=source,
                n_latency_reps=n_latency_reps,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Benchmark failed for '%s': %s", model_type, exc)
            continue

    if not results:
        logger.warning("No models successfully evaluated.")
        return results

    # Write JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Benchmark results written to %s", json_path)

    # Write markdown table
    table = format_results_table(results)
    md_path = os.path.join(output_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# Model Benchmark Results\n\n")
        f.write(table)
        f.write("\n")
    logger.info("Markdown table written to %s", md_path)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(table)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Benchmark all champion models on the OOT test set"
    )
    parser.add_argument("--trans",  required=True, help="Path to raw transaction CSV")
    parser.add_argument("--id",     required=True, help="Path to raw identity CSV")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(VALID_MODELS),
        choices=list(VALID_MODELS),
        help="Model types to benchmark (default: all)",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmark",
        help="Output directory for reports (default: reports/benchmark)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--latency-reps",
        type=int,
        default=200,
        help="Number of single-sample latency measurements per model (default: 200)",
    )
    args = parser.parse_args()

    run_benchmark(
        trans_path=args.trans,
        id_path=args.id,
        model_types=args.models,
        output_dir=args.output,
        config_path=args.config,
        n_latency_reps=args.latency_reps,
    )
