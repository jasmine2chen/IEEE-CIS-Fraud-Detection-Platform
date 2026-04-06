"""Offline batch scoring pipeline.

Loads the @champion model from the MLflow Model Registry, applies the
feature pipeline, scores transactions, and writes predictions to Parquet.

Predictions include: TransactionID, fraud_probability, is_fraud,
score_timestamp, model_version, model_type.

Usage:
    python -m src.batch_score \\
        --trans  data/raw/test_transaction.csv \\
        --id     data/raw/test_identity.csv \\
        --output data/predictions/batch_YYYYMMDD.parquet \\
        --model  xgboost
"""

import argparse
import logging
import os
from typing import Optional
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src import registry
from src.config import load_config
from src.data_prep.data_loader import prepare_data

logger = logging.getLogger(__name__)


def _load_model_artifacts(model_type: str, tracking_uri: str):
    """Try the MLflow registry first; fall back to disk artefacts.

    Returns
    -------
    tuple[pipeline, model, version_str]
    """
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            pipeline, model = registry.load_champion(
                model_type, tracking_uri=tracking_uri
            )
            # Retrieve the version string for the prediction log.
            client = mlflow.MlflowClient()
            model_name = registry.get_model_name(model_type)
            mv = client.get_model_version_by_alias(
                name=model_name, alias="champion"
            )
            version = mv.version
            logger.info(
                "Loaded @champion '%s' version %s from MLflow registry.",
                model_type, version,
            )
            return pipeline, model, version
        except Exception as exc:
            logger.warning(
                "Registry load failed (%s) — falling back to disk artefacts.", exc
            )

    # Disk fall-back
    pipeline = joblib.load("models/feature_pipeline.joblib")
    model = joblib.load("models/xgboost_fraud_model.joblib")
    logger.info("Loaded artefacts from disk.")
    return pipeline, model, "disk"


def run_batch_score(
    trans_path: str,
    id_path: str,
    output_path: str,
    model_type: str = "xgboost",
    config_path: Optional[str] = None,
    include_features: bool = True,
) -> pd.DataFrame:
    """Score a full transaction file and write predictions to Parquet.

    Parameters
    ----------
    trans_path:
        Path to raw transaction CSV.
    id_path:
        Path to raw identity CSV.
    output_path:
        Destination path for the predictions Parquet file.
    model_type:
        Registered model type to use for scoring.
    config_path:
        Path to the YAML config file (uses default if None).
    include_features:
        If True (default), append raw input feature columns to the output
        Parquet so that drift.py can compare feature distributions against
        the reference saved during training.  Pass False to produce a
        compact predictions-only file.

    Returns
    -------
    pd.DataFrame
        The predictions DataFrame that was written to disk.
    """
    cfg = load_config(config_path)
    training_cfg = cfg["training"]
    serving_cfg = cfg["serving"]
    tracking_uri = training_cfg.get("mlflow_tracking_uri", "")
    fraud_threshold = serving_cfg.get("fraud_threshold_prob", 0.5)

    pipeline, model, model_version = _load_model_artifacts(model_type, tracking_uri)

    logger.info("Loading and preparing data from %s + %s ...", trans_path, id_path)
    X, _ = prepare_data(trans_path, id_path)

    logger.info("Applying feature pipeline to %d rows...", len(X))
    X_proc = pipeline.transform(X)

    logger.info("Scoring with model type '%s'...", model_type)
    probs = model.predict_proba(X_proc)[:, 1]

    # Build output DataFrame
    if "TransactionID" in X.columns:
        transaction_ids = X["TransactionID"].values
    else:
        transaction_ids = X.index.values

    predictions = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "fraud_probability": probs.astype(float),
            "is_fraud": (probs >= fraud_threshold).astype(bool),
            "score_timestamp": pd.Timestamp.now(),
            "model_type": model_type,
            "model_version": str(model_version),
        }
    )

    # Optionally append raw input features for downstream drift monitoring.
    # Columns that already exist in predictions (e.g. TransactionID) are skipped.
    if include_features:
        for col in X.columns:
            if col not in predictions.columns:
                predictions[col] = X[col].values

    # Write to Parquet
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out, index=False)
    logger.info(
        "Predictions written to %s  (%d rows)", out, len(predictions)
    )

    # Log batch scoring run to MLflow
    n_scored = len(predictions)
    mean_fraud_prob = float(predictions["fraud_probability"].mean())
    fraud_rate = float(predictions["is_fraud"].mean())

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment("batch_scoring")
        with mlflow.start_run(run_name=f"batch_score_{model_type}"):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_version", str(model_version))
            mlflow.log_param("trans_path", trans_path)
            mlflow.log_param("id_path", id_path)
            mlflow.log_param("output_path", str(out))
            mlflow.log_metric("n_scored", n_scored)
            mlflow.log_metric("mean_fraud_prob", mean_fraud_prob)
            mlflow.log_metric("fraud_rate", fraud_rate)
        logger.info(
            "Batch scoring run logged to MLflow — n_scored=%d, "
            "mean_fraud_prob=%.4f, fraud_rate=%.4f",
            n_scored, mean_fraud_prob, fraud_rate,
        )
    except Exception as exc:
        logger.warning("Failed to log batch scoring run to MLflow: %s", exc)

    return predictions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Offline batch scoring — loads @champion model and scores transactions"
    )
    parser.add_argument("--trans", required=True, help="Path to raw transaction CSV")
    parser.add_argument("--id", required=True, help="Path to raw identity CSV")
    parser.add_argument(
        "--output", required=True, help="Output path for predictions Parquet file"
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(registry.MODEL_NAME_MAP.keys()),
        help="Model type to use for scoring (default: xgboost)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--include-features",
        dest="include_features",
        action="store_true",
        default=True,
        help="Include raw input features in output Parquet for drift monitoring (default: on)",
    )
    parser.add_argument(
        "--no-include-features",
        dest="include_features",
        action="store_false",
        help="Omit raw input features from output Parquet (compact mode)",
    )
    args = parser.parse_args()
    run_batch_score(
        trans_path=args.trans,
        id_path=args.id,
        output_path=args.output,
        model_type=args.model,
        config_path=args.config,
        include_features=args.include_features,
    )
