"""MLflow Model Registry — register, promote, and load models.

Alias convention (MLflow 2.x — stage transitions are deprecated):
    @champion   the model currently serving live production traffic
    @challenger  a candidate model under A/B evaluation before promotion

One registered model per model_type:
    fraud_detection_xgboost
    fraud_detection_mlp_xgboost
    fraud_detection_gnn
    fraud_detection_transformer_xgboost
"""

import logging
from typing import Any, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry configuration
# ---------------------------------------------------------------------------

MODEL_NAME_MAP: dict = {
    "xgboost": "fraud_detection_xgboost",
    "mlp_xgboost": "fraud_detection_mlp_xgboost",
    "gnn": "fraud_detection_gnn",
    "transformer_xgboost": "fraud_detection_transformer_xgboost",
}

# Canonical artifact path logged by train.py for each model type.
# All model-specific training functions log their final XGBoost model to this
# path so the registry can locate it unambiguously regardless of the tuning path.
CANONICAL_XGB_ARTIFACT: dict = {
    "xgboost": "xgboost_model",
    "mlp_xgboost": "mlp_xgboost_final_model",
    "gnn": "gnn_final_model",
    "transformer_xgboost": "transformer_xgboost_final_model",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_model_name(model_type: str) -> str:
    """Validate model_type and return the corresponding registered model name.

    Raises
    ------
    ValueError
        If model_type is not a recognised key in MODEL_NAME_MAP.
    """
    if model_type not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {list(MODEL_NAME_MAP.keys())}"
        )
    return MODEL_NAME_MAP[model_type]


def register_model(
    run_id: str,
    model_type: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[dict] = None,
) -> str:
    """Register the canonical XGBoost artifact from a training run.

    Creates the registered model in the registry if it does not already exist,
    then registers a new version from the specified run.

    Parameters
    ----------
    run_id:
        MLflow run ID that contains the logged model artifact.
    model_type:
        One of ``MODEL_NAME_MAP`` keys — determines both the registered model
        name and the artifact path to register.
    tracking_uri:
        MLflow tracking server URI.  Uses the current default if None.
    tags:
        Optional key/value tags to attach to the registered model version.

    Returns
    -------
    str
        The newly created model version string (e.g. "3").
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = get_model_name(model_type)
    artifact_path = CANONICAL_XGB_ARTIFACT[model_type]
    model_uri = f"runs:/{run_id}/{artifact_path}"

    client = MlflowClient()

    # Create the registered model container if it does not exist yet.
    try:
        client.create_registered_model(
            name=model_name,
            tags={"model_type": model_type},
        )
        logger.info("Created new registered model: %s", model_name)
    except MlflowException as exc:
        # RESOURCE_ALREADY_EXISTS — safe to ignore; any other error re-raised.
        if "already exists" not in str(exc).lower() and "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise

    # Register (create) a new version.
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags=tags or {},
    )
    logger.info(
        "Registered %s version %s from run %s (artifact: %s)",
        model_name, mv.version, run_id, artifact_path,
    )
    return mv.version


def promote_to_champion(
    model_type: str,
    version: str,
    tracking_uri: Optional[str] = None,
) -> None:
    """Set the @champion alias on a specific model version.

    Uses the MLflow 2.x alias API — stage transitions (Staging/Production)
    are deprecated and intentionally not used here.

    Parameters
    ----------
    model_type:
        Registered model to update.
    version:
        Version string to alias as @champion.
    tracking_uri:
        MLflow tracking server URI.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = get_model_name(model_type)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(version),
    )
    logger.info(
        "Set @champion alias → %s version %s", model_name, version
    )


def get_champion_run_id(
    model_type: str,
    tracking_uri: Optional[str] = None,
) -> Optional[str]:
    """Return the MLflow run_id associated with the @champion version.

    Returns None if no @champion alias has been set or if the registered model
    does not exist.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = get_model_name(model_type)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="champion")
        return mv.run_id
    except MlflowException as exc:
        logger.debug(
            "get_champion_run_id: no @champion for %s — %s", model_name, exc
        )
        return None


def load_champion(
    model_type: str,
    tracking_uri: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load the feature pipeline and XGBoost model for the @champion version.

    The feature pipeline is loaded from the champion run's artifact store
    (``runs:/{run_id}/feature_pipeline``).  The XGBoost model is loaded via
    the model alias URI (``models:/{model_name}@champion``), which always
    resolves to the currently aliased version.

    Parameters
    ----------
    model_type:
        Registered model to load.
    tracking_uri:
        MLflow tracking server URI.

    Returns
    -------
    Tuple[pipeline, xgb_model]
        (sklearn Pipeline, XGBoost Booster/classifier)

    Raises
    ------
    MlflowException
        If the @champion alias has not been set for the requested model type.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = get_model_name(model_type)

    # Resolve the champion run_id so we can load the co-versioned pipeline.
    run_id = get_champion_run_id(model_type, tracking_uri=tracking_uri)
    if run_id is None:
        raise MlflowException(
            f"No @champion alias set for registered model '{model_name}'. "
            "Run 'make promote' or call promote_to_champion() first."
        )

    logger.info(
        "Loading @champion for '%s' — run_id=%s", model_type, run_id
    )

    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/feature_pipeline")
    xgb_model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

    return pipeline, xgb_model
