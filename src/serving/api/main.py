import asyncio
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import load_config
from src.serving.registry import load_champion, load_challenger, get_model_name
from src.serving import registry as _registry
from src.serving.api.schemas import (
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

_MODEL_TYPE  = os.getenv("MODEL_TYPE", "xgboost")
_MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "")
_SHADOW_MODE = os.getenv("SHADOW_MODE", "false").lower() == "true"
_API_KEY     = os.getenv("API_KEY", "")

_NEURAL_HYBRIDS = ("mlp_xgboost", "transformer_xgboost", "gnn_xgboost")


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
# Encoder loading helpers
# ---------------------------------------------------------------------------

def _load_encoder_from_disk(model_type: str) -> Optional[object]:
    """Load neural encoder artifact from disk for the given model type.

    Returns the encoder object (MLPEncoder, TabTransformerEncoder, or GNNArtifact)
    or None if artifacts are not present or model is xgboost.
    """
    if model_type not in _NEURAL_HYBRIDS:
        return None

    if model_type == "mlp_xgboost":
        enc_path = Path("models/mlp_xgboost/encoder.pt")
        if not enc_path.exists():
            return None
        import torch
        from src.models.mlp_tree import MLPEncoder
        ckpt = torch.load(str(enc_path), map_location="cpu", weights_only=False)
        enc  = MLPEncoder(
            input_dim=ckpt["input_dim"],
            hidden_dims=tuple(ckpt.get("hidden_dims", [256, 128, 64])),
        )
        enc.load_state_dict(ckpt["model_state_dict"])
        enc.eval()
        return enc

    if model_type == "transformer_xgboost":
        enc_path = Path("models/transformer_xgboost/encoder.pt")
        if not enc_path.exists():
            return None
        import torch
        from src.models.transformer_tree import TabTransformerEncoder
        ckpt = torch.load(str(enc_path), map_location="cpu", weights_only=False)
        enc  = TabTransformerEncoder(
            input_dim=ckpt["input_dim"],
            d_model=ckpt.get("d_model", 64),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 2),
            dim_feedforward=ckpt.get("dim_feedforward", 256),
            dropout=ckpt.get("dropout", 0.1),
        )
        enc.load_state_dict(ckpt["model_state_dict"])
        enc.eval()
        return enc

    # gnn_xgboost
    enc_path = Path("models/gnn_xgboost/encoder.pt")
    h0_path  = Path("models/gnn_xgboost/card_h0_mean.pkl")
    h1_path  = Path("models/gnn_xgboost/card_h1_mean.pkl")
    if not all(p.exists() for p in [enc_path, h0_path, h1_path]):
        return None
    import torch
    from src.models.gnn_tree import GraphSAGEEncoder, GNNArtifact
    ckpt = torch.load(str(enc_path), map_location="cpu", weights_only=False)
    enc  = GraphSAGEEncoder(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt.get("hidden_dim", 64),
        out_dim=ckpt.get("embed_dim", 32),
        dropout=ckpt.get("dropout", 0.1),
    )
    enc.load_state_dict(ckpt["model_state_dict"])
    enc.eval()
    with open(str(h0_path), "rb") as f:
        card_h0_mean = pickle.load(f)
    with open(str(h1_path), "rb") as f:
        card_h1_mean = pickle.load(f)
    return GNNArtifact(encoder=enc, card_h0_mean=card_h0_mean, card_h1_mean=card_h1_mean)


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts for model_type='%s'...", _MODEL_TYPE)
    if _MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            pipeline, model = load_champion(_MODEL_TYPE, tracking_uri=_MLFLOW_URI)
            ml_artifacts["pipeline"] = pipeline
            ml_artifacts["model"]    = model
            ml_artifacts["source"]   = "registry"

            client     = mlflow.MlflowClient()
            model_name = get_model_name(_MODEL_TYPE)
            mv         = client.get_model_version_by_alias(name=model_name, alias="champion")
            ml_artifacts["model_version"] = mv.version

            logger.info("Loaded @champion '%s' version %s from MLflow registry.",
                        _MODEL_TYPE, mv.version)
        except Exception as exc:
            logger.warning("Registry load failed (%s) — falling back to disk.", exc)
            ml_artifacts["pipeline"]      = joblib.load("models/feature_pipeline.joblib")
            ml_artifacts["model"]         = joblib.load("models/xgboost_fraud_model.joblib")
            ml_artifacts["source"]        = "disk"
            ml_artifacts["model_version"] = "disk"
    else:
        ml_artifacts["pipeline"]      = joblib.load("models/feature_pipeline.joblib")
        ml_artifacts["model"]         = joblib.load("models/xgboost_fraud_model.joblib")
        ml_artifacts["source"]        = "disk"
        ml_artifacts["model_version"] = "disk"

    # Load neural encoder if applicable
    encoder = _load_encoder_from_disk(_MODEL_TYPE)
    if encoder is not None:
        ml_artifacts["encoder"] = encoder
        logger.info("Loaded %s encoder from disk.", _MODEL_TYPE)

    if _SHADOW_MODE and _MLFLOW_URI:
        try:
            challenger = load_challenger(_MODEL_TYPE, tracking_uri=_MLFLOW_URI)
            if challenger is not None:
                ml_artifacts["challenger_pipeline"], ml_artifacts["challenger_model"] = challenger

                client     = mlflow.MlflowClient()
                model_name = get_model_name(_MODEL_TYPE)
                ch_mv      = client.get_model_version_by_alias(name=model_name, alias="challenger")
                ml_artifacts["challenger_version"] = ch_mv.version

                ch_encoder = _load_encoder_from_disk(_MODEL_TYPE)
                if ch_encoder is not None:
                    ml_artifacts["challenger_encoder"] = ch_encoder

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

def _build_enriched_input(
    pipeline, encoder_or_artifact, df: pd.DataFrame
) -> np.ndarray:
    """Transform raw request DataFrame → enriched feature matrix.

    For xgboost: pipeline features only.
    For neural hybrids: pipeline features + encoder embeddings.
    GNN uses card1 from the DataFrame for neighbourhood lookup.
    """
    X_proc = pipeline.transform(df)

    if encoder_or_artifact is None:
        return X_proc

    # Determine encoder type and extract embeddings
    from src.models.gnn_tree import GNNArtifact
    if isinstance(encoder_or_artifact, GNNArtifact):
        from src.models.gnn_tree import extract_gnn_embeddings
        card1_values = df["card1"].values if "card1" in df.columns else None
        embeddings = extract_gnn_embeddings(
            encoder_or_artifact, X_proc, card1_values, device="cpu"
        )
    else:
        # MLPEncoder or TabTransformerEncoder — duck-type via embed_dim attribute
        try:
            from src.models.transformer_tree import TabTransformerEncoder
            if isinstance(encoder_or_artifact, TabTransformerEncoder):
                from src.models.transformer_tree import extract_transformer_embeddings
                embeddings = extract_transformer_embeddings(encoder_or_artifact, X_proc, device="cpu")
            else:
                from src.models.mlp_tree import extract_mlp_embeddings
                embeddings = extract_mlp_embeddings(encoder_or_artifact, X_proc, device="cpu")
        except Exception:
            from src.models.mlp_tree import extract_mlp_embeddings
            embeddings = extract_mlp_embeddings(encoder_or_artifact, X_proc, device="cpu")

    return np.hstack([X_proc, embeddings])


def _run_inference(pipeline, model, encoder_or_artifact, request: TransactionRequest) -> tuple:
    """CPU-bound: transform + encode + predict for one transaction.

    Returns (fraud_prob, elapsed_ms). Runs via asyncio.to_thread so it does
    not block the event loop.
    """
    df = pd.DataFrame([request.model_dump(exclude={"transaction_id"})])
    t0 = time.perf_counter()
    X_input = _build_enriched_input(pipeline, encoder_or_artifact, df)
    probs   = model.predict_proba(X_input)
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
        "encoder_loaded": "encoder" in ml_artifacts,
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
            encoder  = artifacts.get("challenger_encoder")
            version  = artifacts.get("challenger_version", "challenger")
            logger.info("Canary routing → @challenger  Amount: $%.2f", request.TransactionAmt)
        else:
            pipeline = artifacts["pipeline"]
            model    = artifacts["model"]
            encoder  = artifacts.get("encoder")
            version  = artifacts.get("model_version", "unknown")
            logger.info("Single prediction request — Amount: $%.2f", request.TransactionAmt)

        fraud_prob, elapsed_ms = await asyncio.to_thread(
            _run_inference, pipeline, model, encoder, request
        )

        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning("Latency budget exceeded: %.1fms > %.1fms (Amount: $%.2f)",
                           elapsed_ms, LATENCY_BUDGET_MS, request.TransactionAmt)

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

        version  = artifacts.get("model_version", "unknown")
        pipeline = artifacts["pipeline"]
        model    = artifacts["model"]
        encoder  = artifacts.get("encoder")

        def _run_batch():
            df = pd.DataFrame([
                req.model_dump(exclude={"transaction_id"}) for req in request.transactions
            ])
            t0      = time.perf_counter()
            X_input = _build_enriched_input(pipeline, encoder, df)
            probs   = model.predict_proba(X_input)
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
    for offline analysis and drift tracking. If no challenger is loaded,
    challenger fields are None.
    """
    try:
        champion_version = artifacts.get("model_version", "unknown")
        champion_pipeline = artifacts["pipeline"]
        champion_model    = artifacts["model"]
        champion_encoder  = artifacts.get("encoder")

        if "challenger_model" in artifacts:
            ch_pipeline = artifacts["challenger_pipeline"]
            ch_model    = artifacts["challenger_model"]
            ch_encoder  = artifacts.get("challenger_encoder")

            champion_result, challenger_result = await asyncio.gather(
                asyncio.to_thread(_run_inference, champion_pipeline, champion_model, champion_encoder, request),
                asyncio.to_thread(_run_inference, ch_pipeline, ch_model, ch_encoder, request),
                return_exceptions=True,
            )
            champion_prob, elapsed_ms = champion_result
            if isinstance(challenger_result, Exception):
                logger.warning("Challenger inference failed in shadow mode: %s", challenger_result)
                challenger_prob = None
            else:
                challenger_prob, ch_ms = challenger_result
                logger.debug("Shadow challenger: prob=%.4f  delta=%.4f  ch_latency=%.1fms",
                             challenger_prob, champion_prob - challenger_prob, ch_ms)
        else:
            champion_prob, elapsed_ms = await asyncio.to_thread(
                _run_inference, champion_pipeline, champion_model, champion_encoder, request
            )
            challenger_prob = None

        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning("Shadow endpoint champion latency: %.1fms > budget %.1fms",
                           elapsed_ms, LATENCY_BUDGET_MS)

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
    uvicorn.run("src.serving.api.main:app", host="0.0.0.0", port=8000, reload=True)
