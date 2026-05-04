"""Data and prediction drift detection using Evidently.

Compares a reference dataset (training distribution) against a current
production window. Drift metrics are logged to MLflow and optionally
persisted to PostgreSQL for Grafana dashboard queries.

Reference data: saved as 'reference_stats/reference.parquet' artifact
during training. Loaded from the @champion run's artifact store.

Usage:
    python -m src.monitoring.drift \\
        --current  data/predictions/batch_YYYYMMDD.parquet \\
        --model    xgboost \\
        --persist  (flag: write metrics to PostgreSQL)
"""

import argparse
import logging
import os
import tempfile
from typing import Optional

import mlflow
import pandas as pd

from src.deployment import registry
from src.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidently import guard
# ---------------------------------------------------------------------------
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.debug(
        "evidently not installed — drift detection will raise if called. "
        "Install with: pip install evidently==0.4.33"
    )


def _require_evidently() -> None:
    if not _EVIDENTLY_AVAILABLE:
        raise ImportError(
            "evidently is required for drift detection but is not installed. "
            "Install it with: pip install evidently==0.4.33"
        )


# ---------------------------------------------------------------------------
# Core drift functions
# ---------------------------------------------------------------------------

def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping=None,
) -> dict:
    """Compute data drift and data quality metrics with Evidently.

    Parameters
    ----------
    reference_df:
        Training/reference distribution DataFrame.
    current_df:
        Current production window DataFrame.
    column_mapping:
        Optional Evidently ColumnMapping for target/prediction columns.

    Returns
    -------
    dict with keys:
        dataset_drift_detected: bool
        n_drifted_features: int
        share_drifted_features: float
        per_feature_drift: dict mapping feature name → drift detected bool
    """
    _require_evidently()

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    # Parse DataDrift results
    drift_results = {}
    dataset_drift_detected = False
    n_drifted_features = 0
    share_drifted_features = 0.0
    per_feature_drift: dict = {}

    for metric_result in result.get("metrics", []):
        metric_id = metric_result.get("metric", "")
        if "DatasetDriftMetric" in metric_id or metric_result.get("metric") == "DatasetDriftMetric":
            drift_results = metric_result.get("result", {})
            dataset_drift_detected = bool(drift_results.get("dataset_drift", False))
            n_drifted_features = int(drift_results.get("number_of_drifted_columns", 0))
            share_drifted_features = float(drift_results.get("share_of_drifted_columns", 0.0))

        if "ColumnDriftMetric" in metric_id or metric_result.get("metric") == "ColumnDriftMetric":
            col_result = metric_result.get("result", {})
            col_name = col_result.get("column_name", "unknown")
            per_feature_drift[col_name] = bool(col_result.get("drift_detected", False))

    return {
        "dataset_drift_detected": dataset_drift_detected,
        "n_drifted_features": n_drifted_features,
        "share_drifted_features": share_drifted_features,
        "per_feature_drift": per_feature_drift,
    }


def compute_prediction_drift(
    reference_probs: pd.Series,
    current_probs: pd.Series,
) -> dict:
    """Compute prediction drift using Evidently TargetDriftPreset.

    Parameters
    ----------
    reference_probs:
        Series of fraud probability scores from the reference (training) period.
    current_probs:
        Series of fraud probability scores from the current scoring window.

    Returns
    -------
    dict with keys:
        drift_detected: bool
        drift_score: float  (KS statistic or similar distance)
        mean_reference: float
        mean_current: float
    """
    _require_evidently()

    ref_df = pd.DataFrame({"prediction": reference_probs.values})
    cur_df = pd.DataFrame({"prediction": current_probs.values})

    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()

    drift_detected = False
    drift_score = 0.0

    for metric_result in result.get("metrics", []):
        metric_id = metric_result.get("metric", "")
        if "ColumnDriftMetric" in metric_id or "TargetDrift" in metric_id:
            col_result = metric_result.get("result", {})
            drift_detected = bool(col_result.get("drift_detected", False))
            drift_score = float(
                col_result.get("drift_score", col_result.get("stattest_threshold", 0.0))
            )
            break

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "mean_reference": float(reference_probs.mean()),
        "mean_current": float(current_probs.mean()),
    }


def load_reference_data(
    model_type: str,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """Load the reference dataset from the @champion run's artifact store.

    The reference Parquet is expected at artifact path
    ``reference_stats/reference.parquet`` within the champion run.

    Falls back gracefully with a warning if the artifact is not found,
    returning an empty DataFrame.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = registry.get_champion_run_id(model_type, tracking_uri=tracking_uri)
    if run_id is None:
        logger.warning(
            "No @champion run found for '%s' — returning empty reference DataFrame.",
            model_type,
        )
        return pd.DataFrame()

    client = mlflow.MlflowClient()
    artifact_path = "reference_stats/reference.parquet"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(
                run_id=run_id,
                path=artifact_path,
                dst_path=tmp_dir,
            )
            ref_df = pd.read_parquet(local_path)
            logger.info(
                "Loaded reference data from run %s: %d rows, %d columns",
                run_id, len(ref_df), ref_df.shape[1],
            )
            return ref_df
    except Exception as exc:
        logger.warning(
            "Could not load reference data from run %s (artifact: %s): %s — "
            "returning empty DataFrame.",
            run_id, artifact_path, exc,
        )
        return pd.DataFrame()


def persist_to_postgres(metrics: dict, db_url: str) -> None:
    """Insert a drift metrics row into the PostgreSQL ``drift_metrics`` table.

    Creates the table if it does not already exist.

    Parameters
    ----------
    metrics:
        Dict with at minimum the keys produced by run_drift_check().
    db_url:
        PostgreSQL connection string
        (e.g. ``postgresql://user:pass@host:5432/dbname``).
    """
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2-binary is required to persist metrics to PostgreSQL. "
            "Install with: pip install psycopg2-binary==2.9.9"
        ) from exc

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS drift_metrics (
        id                      SERIAL PRIMARY KEY,
        model_type              VARCHAR(50)  NOT NULL,
        model_version           VARCHAR(20),
        evaluation_timestamp    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        n_samples               INTEGER,
        dataset_drift_detected  BOOLEAN,
        n_drifted_features      INTEGER,
        share_drifted_features  FLOAT,
        prediction_drift_score  FLOAT,
        prediction_drift_detected BOOLEAN,
        mean_fraud_prob         FLOAT
    );
    """

    insert_sql = """
    INSERT INTO drift_metrics (
        model_type,
        model_version,
        evaluation_timestamp,
        n_samples,
        dataset_drift_detected,
        n_drifted_features,
        share_drifted_features,
        prediction_drift_score,
        prediction_drift_detected,
        mean_fraud_prob
    ) VALUES (
        %(model_type)s,
        %(model_version)s,
        NOW(),
        %(n_samples)s,
        %(dataset_drift_detected)s,
        %(n_drifted_features)s,
        %(share_drifted_features)s,
        %(prediction_drift_score)s,
        %(prediction_drift_detected)s,
        %(mean_fraud_prob)s
    );
    """

    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                cur.execute(
                    insert_sql,
                    {
                        "model_type": metrics.get("model_type"),
                        "model_version": metrics.get("model_version"),
                        "n_samples": metrics.get("n_samples"),
                        "dataset_drift_detected": metrics.get("dataset_drift_detected"),
                        "n_drifted_features": metrics.get("n_drifted_features"),
                        "share_drifted_features": metrics.get("share_drifted_features"),
                        "prediction_drift_score": metrics.get("prediction_drift_score"),
                        "prediction_drift_detected": metrics.get("prediction_drift_detected"),
                        "mean_fraud_prob": metrics.get("mean_fraud_prob"),
                    },
                )
        logger.info("Drift metrics persisted to PostgreSQL.")
    finally:
        conn.close()


def run_drift_check(
    current_path: str,
    model_type: str,
    tracking_uri: str,
    db_url: Optional[str] = None,
) -> dict:
    """Orchestrate the full drift check pipeline.

    Steps:
    1. Load the current predictions/scoring batch from Parquet.
    2. Load the reference data from the @champion run's artifact store.
    3. Compute dataset drift and prediction drift (if evidently available).
    4. Log all metrics to MLflow under a "drift_monitoring" experiment.
    5. Optionally persist metrics to PostgreSQL.

    Parameters
    ----------
    current_path:
        Path to the current scoring batch Parquet (output of batch_score.py).
    model_type:
        Registered model type being monitored.
    tracking_uri:
        MLflow tracking server URI.
    db_url:
        PostgreSQL connection URL for persistence (skip if None).

    Returns
    -------
    dict
        Combined metrics from both drift checks plus metadata.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Load current batch
    current_df = pd.read_parquet(current_path)
    logger.info(
        "Loaded current batch from %s: %d rows", current_path, len(current_df)
    )

    # Load reference
    reference_df = load_reference_data(model_type, tracking_uri=tracking_uri)

    # Resolve champion version for tagging
    model_version = "unknown"
    try:
        client = mlflow.MlflowClient()
        model_name = registry.get_model_name(model_type)
        mv = client.get_model_version_by_alias(name=model_name, alias="champion")
        model_version = mv.version
    except Exception:
        pass

    metrics: dict = {
        "model_type": model_type,
        "model_version": model_version,
        "n_samples": len(current_df),
        "mean_fraud_prob": float(
            current_df["fraud_probability"].mean()
            if "fraud_probability" in current_df.columns
            else 0.0
        ),
    }

    # Dataset drift
    if not reference_df.empty and _EVIDENTLY_AVAILABLE:
        try:
            # Align columns — only score features present in both
            feature_cols = [
                c for c in reference_df.columns
                if c in current_df.columns
                and c not in ("TransactionID", "score_timestamp", "model_type", "model_version")
            ]
            if feature_cols:
                drift_result = compute_drift_report(
                    reference_df[feature_cols],
                    current_df[feature_cols],
                )
                metrics.update(drift_result)
            else:
                logger.warning("No overlapping feature columns for drift comparison.")
                metrics.update({
                    "dataset_drift_detected": None,
                    "n_drifted_features": None,
                    "share_drifted_features": None,
                    "per_feature_drift": {},
                })
        except Exception as exc:
            logger.warning("Dataset drift computation failed: %s", exc)
            metrics.update({
                "dataset_drift_detected": None,
                "n_drifted_features": None,
                "share_drifted_features": None,
                "per_feature_drift": {},
            })
    else:
        metrics.update({
            "dataset_drift_detected": None,
            "n_drifted_features": None,
            "share_drifted_features": None,
            "per_feature_drift": {},
        })

    # Prediction drift
    pred_drift_metrics = {
        "prediction_drift_detected": None,
        "prediction_drift_score": None,
    }
    if (
        not reference_df.empty
        and "fraud_probability" in reference_df.columns
        and "fraud_probability" in current_df.columns
        and _EVIDENTLY_AVAILABLE
    ):
        try:
            pd_result = compute_prediction_drift(
                reference_df["fraud_probability"],
                current_df["fraud_probability"],
            )
            pred_drift_metrics = {
                "prediction_drift_detected": pd_result["drift_detected"],
                "prediction_drift_score": pd_result["drift_score"],
            }
            metrics["mean_reference_fraud_prob"] = pd_result["mean_reference"]
        except Exception as exc:
            logger.warning("Prediction drift computation failed: %s", exc)

    metrics.update(pred_drift_metrics)

    # Log to MLflow
    try:
        mlflow.set_experiment("drift_monitoring")
        with mlflow.start_run(run_name=f"drift_{model_type}"):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_version", str(model_version))
            mlflow.log_param("current_path", current_path)
            mlflow.log_metric("n_samples", metrics["n_samples"])
            mlflow.log_metric("mean_fraud_prob", metrics["mean_fraud_prob"])

            if metrics.get("n_drifted_features") is not None:
                mlflow.log_metric("n_drifted_features", metrics["n_drifted_features"])
                mlflow.log_metric(
                    "share_drifted_features", metrics["share_drifted_features"]
                )
                mlflow.log_metric(
                    "dataset_drift_detected",
                    int(bool(metrics["dataset_drift_detected"])),
                )

            if metrics.get("prediction_drift_score") is not None:
                mlflow.log_metric(
                    "prediction_drift_score", metrics["prediction_drift_score"]
                )
                mlflow.log_metric(
                    "prediction_drift_detected",
                    int(bool(metrics["prediction_drift_detected"])),
                )

        logger.info("Drift metrics logged to MLflow experiment 'drift_monitoring'.")
    except Exception as exc:
        logger.warning("Failed to log drift metrics to MLflow: %s", exc)

    # Persist to PostgreSQL
    if db_url:
        try:
            persist_to_postgres(metrics, db_url)
        except Exception as exc:
            logger.warning("Failed to persist drift metrics to PostgreSQL: %s", exc)

    logger.info(
        "Drift check complete — dataset_drift=%s, n_drifted_features=%s, "
        "prediction_drift=%s",
        metrics.get("dataset_drift_detected"),
        metrics.get("n_drifted_features"),
        metrics.get("prediction_drift_detected"),
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run drift detection against the current scoring batch"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current predictions Parquet (output of batch_score.py)",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(registry.MODEL_NAME_MAP.keys()),
        help="Model type to monitor (default: xgboost)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write drift metrics to PostgreSQL (requires --db-url or POSTGRES_URL env var)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL connection URL (falls back to POSTGRES_URL env var)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/model_config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracking_uri = cfg["training"].get("mlflow_tracking_uri", "")

    db_url = None
    if args.persist:
        db_url = args.db_url or os.getenv("POSTGRES_URL")
        if not db_url:
            logger.error(
                "--persist requires a database URL via --db-url or POSTGRES_URL env var"
            )
            raise SystemExit(1)

    result = run_drift_check(
        current_path=args.current,
        model_type=args.model,
        tracking_uri=tracking_uri,
        db_url=db_url,
    )

    print("\nDrift check results:")
    for k, v in result.items():
        if k != "per_feature_drift":
            print(f"  {k}: {v}")
    if result.get("per_feature_drift"):
        drifted = [f for f, d in result["per_feature_drift"].items() if d]
        print(f"  drifted_features ({len(drifted)}): {drifted[:10]}")
