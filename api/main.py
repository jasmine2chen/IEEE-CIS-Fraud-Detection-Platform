import logging
import os
from contextlib import asynccontextmanager

import joblib
import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.config import load_config
from src.registry import load_champion, get_model_name
from src import registry as _registry
from api.schemas import TransactionRequest, BatchTransactionRequest, PredictionResponse, BatchPredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fraud_api")

ml_artifacts = {}

# Threshold comes from config; never hardcoded here.
_cfg = load_config()
FRAUD_THRESHOLD: float = _cfg["serving"]["fraud_threshold_prob"]

# CORS origins — comma-separated env var for production, wildcard for local dev.
# Example: ALLOWED_ORIGINS="https://app.example.com,https://admin.example.com"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

_MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup; release on shutdown.

    Tries the MLflow Model Registry first (if MLFLOW_TRACKING_URI is set),
    then falls back to disk artefacts if the registry is unavailable.
    """
    logger.info("Loading ML artifacts...")
    if _MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            pipeline, model = load_champion(_MODEL_TYPE, tracking_uri=_MLFLOW_URI)
            ml_artifacts["pipeline"] = pipeline
            ml_artifacts["model"] = model
            ml_artifacts["source"] = "registry"
            logger.info("Loaded @champion '%s' from MLflow registry.", _MODEL_TYPE)
        except Exception as exc:
            logger.warning("Registry load failed (%s) — falling back to disk.", exc)
            ml_artifacts["pipeline"] = joblib.load("models/feature_pipeline.joblib")
            ml_artifacts["model"]    = joblib.load("models/xgboost_fraud_model.joblib")
            ml_artifacts["source"]   = "disk"
    else:
        ml_artifacts["pipeline"] = joblib.load("models/feature_pipeline.joblib")
        ml_artifacts["model"]    = joblib.load("models/xgboost_fraud_model.joblib")
        ml_artifacts["source"]   = "disk"
    logger.info("Artifacts loaded from %s. Threshold: %.4f",
                ml_artifacts["source"], FRAUD_THRESHOLD)
    yield
    logger.info("Shutting down.")
    ml_artifacts.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Production API for serving real-time IEEE-CIS fraud detection predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_prediction_artifacts():
    """Dependency injector for model artifacts."""
    return ml_artifacts


@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "artifacts_loaded": len(ml_artifacts) > 0,
        "model_type": _MODEL_TYPE,
        "artifact_source": ml_artifacts.get("source", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: TransactionRequest,
    artifacts: dict = Depends(get_prediction_artifacts),
):
    """Predict fraud probability for a single transaction."""
    try:
        df = pd.DataFrame([request.model_dump()])
        logger.info("Single prediction request — Amount: $%.2f", request.TransactionAmt)

        X_processed = artifacts["pipeline"].transform(df)
        probs = artifacts["model"].predict_proba(X_processed)
        fraud_prob = float(probs[0, 1])
        is_fraud = fraud_prob >= FRAUD_THRESHOLD

        logger.info("Fraud probability: %.4f  is_fraud: %s", fraud_prob, is_fraud)
        return PredictionResponse(fraud_probability=fraud_prob, is_fraud=is_fraud)
    except Exception as e:
        logger.error("Error during single prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchTransactionRequest,
    artifacts: dict = Depends(get_prediction_artifacts),
):
    """Predict fraud probabilities for a batch of transactions."""
    try:
        batch_size = len(request.transactions)
        logger.info("Batch prediction request — %d transactions.", batch_size)

        df = pd.DataFrame([req.model_dump() for req in request.transactions])
        X_processed = artifacts["pipeline"].transform(df)
        probs = artifacts["model"].predict_proba(X_processed)
        fraud_probs = probs[:, 1].astype(float)

        responses = [
            PredictionResponse(
                fraud_probability=float(prob),
                is_fraud=float(prob) >= FRAUD_THRESHOLD,
            )
            for prob in fraud_probs
        ]
        logger.info("Batch of %d processed successfully.", batch_size)
        return BatchPredictionResponse(predictions=responses)
    except Exception as e:
        logger.error("Error during batch prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
