"""Prefect orchestration flow for the full fraud detection training pipeline.

Chains: tune → train → register → evaluate → promote to @champion.

Can be run directly (no Prefect server required) or deployed to
Prefect Cloud / a self-hosted Prefect server for scheduling.

Usage (local):
    python pipelines/training_pipeline.py --model xgboost --trials 50

Usage (Prefect deployment):
    prefect deploy pipelines/training_pipeline.py:training_pipeline
"""

import argparse
import logging
from typing import Optional

import mlflow
from prefect import flow, task, get_run_logger

from src.deployment import registry
from src.config import load_config
from src.training.train import train
from src.training.tune import run_tuning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="tune-hyperparameters", retries=1, retry_delay_seconds=30)
def tune_task(
    trans_path: str,
    id_path: str,
    config_path: Optional[str],
    model_type: str,
    n_trials: int,
) -> str:
    """Run Optuna HPO + RFE and write best params back to model_config.yaml.

    Returns a human-readable summary string of the best parameters found.
    """
    task_logger = get_run_logger()
    task_logger.info(
        "Starting hyperparameter tuning — model=%s, trials=%d", model_type, n_trials
    )
    run_tuning(
        trans_path=trans_path,
        id_path=id_path,
        config_path=config_path,
        model_type=model_type,
        n_trials=n_trials,
    )
    summary = f"Tuning complete — model={model_type}, n_trials={n_trials}"
    task_logger.info(summary)
    return summary


@task(name="train-model", retries=1, retry_delay_seconds=30)
def train_task(
    trans_path: str,
    id_path: str,
    config_path: Optional[str],
    model_type: str,
) -> str:
    """Train the model and return the MLflow run_id.

    Calls src.train.train() which starts an MLflow run, trains the model,
    logs artefacts, and registers the model.
    """
    task_logger = get_run_logger()
    task_logger.info("Starting model training — model=%s", model_type)

    train(
        trans_path=trans_path,
        id_path=id_path,
        config_path=config_path,
        model_type=model_type,
    )

    run_id = mlflow.last_active_run().info.run_id
    task_logger.info("Training complete — run_id=%s", run_id)
    return run_id


@task(name="register-model")
def register_task(
    run_id: str,
    model_type: str,
    tracking_uri: str,
) -> str:
    """Register the trained model artefact in the MLflow Model Registry.

    Returns the new model version string.
    """
    task_logger = get_run_logger()
    task_logger.info(
        "Registering model — model_type=%s, run_id=%s", model_type, run_id
    )
    version = registry.register_model(
        run_id=run_id,
        model_type=model_type,
        tracking_uri=tracking_uri,
    )
    task_logger.info(
        "Registered %s version %s", registry.get_model_name(model_type), version
    )
    return version


@task(name="evaluate-stability")
def evaluate_task(run_id: str, tracking_uri: str) -> bool:
    """Check whether the training run passed the stability gate.

    The stability gate sets the MLflow tag ``stability_gate_passed = "True"``
    when ``var(fold FPRs) < 0.03``. Returns True if the gate passed.
    """
    task_logger = get_run_logger()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    gate_passed_tag = run.data.tags.get("stability_gate_passed", "False")
    stability_passed = gate_passed_tag == "True"

    task_logger.info(
        "Stability gate for run %s: %s (tag value: %r)",
        run_id, "PASSED" if stability_passed else "FAILED", gate_passed_tag,
    )
    return stability_passed


@task(name="promote-champion")
def promote_task(
    model_type: str,
    version: str,
    tracking_uri: str,
) -> None:
    """Set the @champion alias on the registered model version."""
    task_logger = get_run_logger()
    task_logger.info(
        "Promoting %s version %s to @champion",
        registry.get_model_name(model_type), version,
    )
    registry.promote_to_champion(
        model_type=model_type,
        version=version,
        tracking_uri=tracking_uri,
    )
    task_logger.info(
        "@champion alias set — %s v%s is now live.",
        registry.get_model_name(model_type), version,
    )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="fraud-detection-training-pipeline", log_prints=True)
def training_pipeline(
    trans_path: str = "data/raw/train_transaction.csv",
    id_path: str = "data/raw/train_identity.csv",
    config_path: Optional[str] = None,
    model_type: str = "xgboost",
    n_trials: int = 50,
    auto_promote: bool = True,
) -> dict:
    """Full training pipeline: tune → train → register → evaluate → promote.

    Parameters
    ----------
    trans_path:
        Path to raw transaction CSV.
    id_path:
        Path to raw identity CSV.
    config_path:
        Path to YAML config (uses default if None).
    model_type:
        Model type to train (xgboost, mlp_xgboost, transformer_xgboost, gnn_xgboost).
    n_trials:
        Number of Optuna trials for the tuning step.
    auto_promote:
        If True and the stability gate passes, promote to @champion.

    Returns
    -------
    dict with keys:
        run_id, version, stability_passed, promoted
    """
    flow_logger = get_run_logger()
    flow_logger.info(
        "Starting fraud-detection-training-pipeline — model=%s, n_trials=%d",
        model_type, n_trials,
    )

    # 1. Load config to get tracking_uri
    cfg = load_config(config_path)
    tracking_uri: str = cfg["training"].get("mlflow_tracking_uri", "")

    # 2. Tune
    tune_summary = tune_task(
        trans_path=trans_path,
        id_path=id_path,
        config_path=config_path,
        model_type=model_type,
        n_trials=n_trials,
    )
    flow_logger.info("Tune task result: %s", tune_summary)

    # 3. Train
    run_id = train_task(
        trans_path=trans_path,
        id_path=id_path,
        config_path=config_path,
        model_type=model_type,
    )

    # 4. Register
    version = register_task(
        run_id=run_id,
        model_type=model_type,
        tracking_uri=tracking_uri,
    )

    # 5. Evaluate stability gate
    stability_passed = evaluate_task(run_id=run_id, tracking_uri=tracking_uri)

    # 6. Conditional promotion
    promoted = False
    if auto_promote and stability_passed:
        promote_task(
            model_type=model_type,
            version=version,
            tracking_uri=tracking_uri,
        )
        promoted = True
        flow_logger.info(
            "Promoted to @champion — %s version %s",
            registry.get_model_name(model_type), version,
        )
    else:
        if not stability_passed:
            flow_logger.warning(
                "Stability gate FAILED for run %s — not promoting to @champion. "
                "Review var(fold FPRs) in the MLflow run before manually promoting.",
                run_id,
            )
        else:
            flow_logger.info(
                "auto_promote=False — skipping @champion promotion for version %s.",
                version,
            )

    result = {
        "run_id": run_id,
        "version": version,
        "stability_passed": stability_passed,
        "promoted": promoted,
    }
    flow_logger.info("Pipeline complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the full Prefect fraud-detection training pipeline locally"
    )
    parser.add_argument(
        "--trans",
        default="data/raw/train_transaction.csv",
        help="Path to raw transaction CSV",
    )
    parser.add_argument(
        "--id",
        default="data/raw/train_identity.csv",
        help="Path to raw identity CSV",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(registry.MODEL_NAME_MAP.keys()),
        help="Model type to train (default: xgboost)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna HPO trials (default: 50)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip automatic @champion promotion even if stability gate passes",
    )
    args = parser.parse_args()

    result = training_pipeline(
        trans_path=args.trans,
        id_path=args.id,
        config_path=args.config,
        model_type=args.model,
        n_trials=args.trials,
        auto_promote=not args.no_promote,
    )

    print("\nPipeline result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
