import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import load_config
from src.deployment.registry import load_champion, load_challenger, get_model_name
from src.deployment import registry as _registry
from src.deployment.api.schemas import (
    TransactionRequest,
    BatchTransactionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ShadowPredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fraud_api")

ml_artifacts: dict = {}

_cfg = load_config()
FRAUD_THRESHOLD: float = _cfg["serving"]["fraud_threshold_prob"]

LATENCY_BUDGET_MS: float = float(os.getenv("LATENCY_BUDGET_MS", "100"))

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

_MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
_SHADOW_MODE = os.getenv("SHADOW_MODE", "false").lower() == "true"

# API key auth — set API_KEY env var to enable. Unset = auth disabled (dev mode).
_API_KEY = os.getenv("API_KEY", "")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests with a missing/wrong key when API_KEY is configured."""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts...")
    if _MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            pipeline, model = load_champion(_MODEL_TYPE, tracking_uri=_MLFLOW_URI)
            ml_artifacts["pipeline"] = pipeline
            ml_artifacts["model"] = model
            ml_artifacts["source"] = "registry"

            # Capture the actual registered version for response traceability.
            client = mlflow.MlflowClient()
            model_name = get_model_name(_MODEL_TYPE)
            mv = client.get_model_version_by_alias(name=model_name, alias="champion")
            ml_artifacts["model_version"] = mv.version

            logger.info("Loaded @champion '%s' version %s from MLflow registry.",
                        _MODEL_TYPE, mv.version)
        except Exception as exc:
            logger.warning("Registry load failed (%s) — falling back to disk.", exc)
            ml_artifacts["pipeline"] = joblib.load("models/feature_pipeline.joblib")
            ml_artifacts["model"]    = joblib.load("models/xgboost_fraud_model.joblib")
            ml_artifacts["source"]   = "disk"
            ml_artifacts["model_version"] = "disk"
    else:
        ml_artifacts["pipeline"] = joblib.load("models/feature_pipeline.joblib")
        ml_artifacts["model"]    = joblib.load("models/xgboost_fraud_model.joblib")
        ml_artifacts["source"]   = "disk"
        ml_artifacts["model_version"] = "disk"

    if _SHADOW_MODE and _MLFLOW_URI:
        try:
            challenger = load_challenger(_MODEL_TYPE, tracking_uri=_MLFLOW_URI)
            if challenger is not None:
                ml_artifacts["challenger_pipeline"], ml_artifacts["challenger_model"] = challenger

                client = mlflow.MlflowClient()
                model_name = get_model_name(_MODEL_TYPE)
                ch_mv = client.get_model_version_by_alias(name=model_name, alias="challenger")
                ml_artifacts["challenger_version"] = ch_mv.version

                logger.info("Shadow mode: @challenger version %s loaded for '%s'.",
                            ch_mv.version, _MODEL_TYPE)
            else:
                logger.info("Shadow mode enabled but no @challenger alias set — skipping.")
        except Exception as exc:
            logger.warning("Challenger load failed (%s) — shadow mode disabled.", exc)

    logger.info("Artifacts loaded from %s. Threshold: %.4f",
                ml_artifacts["source"], FRAUD_THRESHOLD)
    yield
    logger.info("Shutting down.")
    ml_artifacts.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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

# Prometheus metrics — scraped at /metrics by Prometheus/Grafana.
Instrumentator().instrument(app).expose(app)


def get_prediction_artifacts():
    """Dependency injector for model artifacts."""
    return ml_artifacts


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(pipeline, model, request: TransactionRequest) -> tuple[float, float]:
    """CPU-bound: run pipeline + model for one transaction.

    Returns (fraud_prob, elapsed_ms). Intended to be called via asyncio.to_thread
    so it does not block the event loop.
    """
    df = pd.DataFrame([request.model_dump(exclude={"transaction_id"})])
    t0 = time.perf_counter()
    X_processed = pipeline.transform(df)
    probs = model.predict_proba(X_processed)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return float(probs[0, 1]), elapsed_ms


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": len(ml_artifacts) > 0,
        "model_type": _MODEL_TYPE,
        "model_version": ml_artifacts.get("model_version", "unknown"),
        "artifact_source": ml_artifacts.get("source", "unknown"),
        "shadow_mode": "challenger_model" in ml_artifacts,
        "challenger_version": ml_artifacts.get("challenger_version"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: TransactionRequest,
    artifacts: dict = Depends(get_prediction_artifacts),
    _: None = Depends(verify_api_key),
    x_canary: Optional[str] = Header(default=None, alias="X-Canary"),
):
    """Predict fraud probability for a single transaction.

    Canary routing: send ``X-Canary: true`` to route to the @challenger model
    (if loaded). The response shape is identical; identify canary responses
    via the ``model_version`` field.
    """
    try:
        use_challenger = (
            x_canary is not None
            and x_canary.lower() == "true"
            and "challenger_model" in artifacts
        )

        if use_challenger:
            pipeline = artifacts["challenger_pipeline"]
            model    = artifacts["challenger_model"]
            version  = artifacts.get("challenger_version", "challenger")
            logger.info("Canary routing → @challenger  Amount: $%.2f", request.TransactionAmt)
        else:
            pipeline = artifacts["pipeline"]
            model    = artifacts["model"]
            version  = artifacts.get("model_version", "unknown")
            logger.info("Single prediction request — Amount: $%.2f", request.TransactionAmt)

        # Run blocking inference off the event loop.
        fraud_prob, elapsed_ms = await asyncio.to_thread(_run_inference, pipeline, model, request)

        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning(
                "Latency budget exceeded: %.1fms > %.1fms (Amount: $%.2f)",
                elapsed_ms, LATENCY_BUDGET_MS, request.TransactionAmt,
            )

        is_fraud = fraud_prob >= FRAUD_THRESHOLD
        logger.info("Fraud probability: %.4f  is_fraud: %s  latency: %.1fms  version: %s",
                    fraud_prob, is_fraud, elapsed_ms, version)

        return PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=fraud_prob,
            is_fraud=is_fraud,
            model_version=version,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during single prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchTransactionRequest,
    artifacts: dict = Depends(get_prediction_artifacts),
    _: None = Depends(verify_api_key),
):
    """Predict fraud probabilities for a batch of transactions."""
    try:
        batch_size = len(request.transactions)
        logger.info("Batch prediction request — %d transactions.", batch_size)

        version = artifacts.get("model_version", "unknown")

        def _run_batch():
            df = pd.DataFrame([
                req.model_dump(exclude={"transaction_id"}) for req in request.transactions
            ])
            t0 = time.perf_counter()
            X_processed = artifacts["pipeline"].transform(df)
            probs = artifacts["model"].predict_proba(X_processed)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return probs[:, 1].astype(float), elapsed_ms

        fraud_probs, elapsed_ms = await asyncio.to_thread(_run_batch)

        per_txn_ms = elapsed_ms / batch_size if batch_size > 0 else elapsed_ms
        if per_txn_ms > LATENCY_BUDGET_MS:
            logger.warning(
                "Batch latency budget exceeded: %.1fms/txn > %.1fms budget (%d txns total)",
                per_txn_ms, LATENCY_BUDGET_MS, batch_size,
            )

        responses = [
            PredictionResponse(
                transaction_id=req.transaction_id,
                fraud_probability=float(prob),
                is_fraud=float(prob) >= FRAUD_THRESHOLD,
                model_version=version,
            )
            for req, prob in zip(request.transactions, fraud_probs)
        ]
        logger.info("Batch of %d processed in %.1fms.", batch_size, elapsed_ms)
        return BatchPredictionResponse(predictions=responses)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during batch prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/shadow", response_model=ShadowPredictionResponse, tags=["Prediction"])
async def predict_shadow(
    request: TransactionRequest,
    artifacts: dict = Depends(get_prediction_artifacts),
    _: None = Depends(verify_api_key),
):
    """Score with both champion and challenger concurrently.

    Returns the champion prediction. The challenger probability is included
    for offline analysis and drift tracking.  If no challenger is loaded,
    challenger fields are None.
    """
    try:
        champion_version = artifacts.get("model_version", "unknown")

        # Run champion and challenger concurrently if challenger is available.
        if "challenger_model" in artifacts:
            challenger_version = artifacts.get("challenger_version", "challenger")
            champion_result, challenger_result = await asyncio.gather(
                asyncio.to_thread(_run_inference, artifacts["pipeline"], artifacts["model"], request),
                asyncio.to_thread(_run_inference, artifacts["challenger_pipeline"], artifacts["challenger_model"], request),
                return_exceptions=True,
            )
            champion_prob, elapsed_ms = champion_result
            if isinstance(challenger_result, Exception):
                logger.warning("Challenger inference failed in shadow mode: %s", challenger_result)
                challenger_prob = None
            else:
                challenger_prob, ch_ms = challenger_result
                logger.debug(
                    "Shadow challenger: prob=%.4f  delta=%.4f  ch_latency=%.1fms",
                    challenger_prob, champion_prob - challenger_prob, ch_ms,
                )
        else:
            champion_prob, elapsed_ms = await asyncio.to_thread(
                _run_inference, artifacts["pipeline"], artifacts["model"], request
            )
            challenger_prob = None
            challenger_version = None

        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning(
                "Shadow endpoint champion latency: %.1fms > budget %.1fms",
                elapsed_ms, LATENCY_BUDGET_MS,
            )

        champion_response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=champion_prob,
            is_fraud=champion_prob >= FRAUD_THRESHOLD,
            model_version=champion_version,
        )

        delta = (
            round(champion_prob - challenger_prob, 6)
            if challenger_prob is not None else None
        )

        return ShadowPredictionResponse(
            champion=champion_response,
            challenger_fraud_probability=challenger_prob,
            champion_challenger_delta=delta,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during shadow prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.deployment.api.main:app", host="0.0.0.0", port=8000, reload=True)
