import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import joblib
from typing import List

from api.schemas import TransactionRequest, BatchTransactionRequest, PredictionResponse, BatchPredictionResponse

# Setup Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fraud_api")

ml_artifacts = {}
FRAUD_THRESHOLD = 0.85 # High precision threshold setting

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Loads the ML models into memory exactly once at application startup.
    Cleans up resources cleanly at shutdown.
    """
    logger.info("Starting up FastAPI application...")
    try:
        logger.info("Loading ML artifacts from disk...")
        ml_artifacts["pipeline"] = joblib.load("models/feature_pipeline.joblib")
        ml_artifacts["model"] = joblib.load("models/xgboost_fraud_model.joblib")
        logger.info("Successfully loaded ML models and preprocessing pipeline.")
    except Exception as e:
        logger.error(f"Failed to load ML artifacts: {e}")
        logger.warning("Failing over to MockModel for testing/development purposes.")
        class MockModel:
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                return np.random.uniform(0, 1, size=(len(X), 2))
        ml_artifacts["model"] = MockModel()
        ml_artifacts["pipeline"] = None
        
    yield # Server is now running and accepting requests
    
    # Clean up on shutdown
    logger.info("Shutting down application...")
    ml_artifacts.clear()
    logger.info("Cleared ML models from memory.")


# Initialize FastAPI App
app = FastAPI(
    title="Fraud Detection API",
    description="Production API for serving real-time IEEE-CIS fraud detection predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"status": "healthy", "artifacts_loaded": len(ml_artifacts) > 0}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: TransactionRequest, artifacts: dict = Depends(get_prediction_artifacts)):
    """
    Predicts fraud probability for a single transaction.
    """
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.model_dump()])
        logger.info(f"Received single prediction request for Amount: ${request.TransactionAmt}")
        
        pipeline = artifacts.get("pipeline")
        model = artifacts.get("model")
        
        if pipeline is not None:
            # Apply proper scikit-learn Preprocessing Pipeline
            X_processed = pipeline.transform(df)
        else:
            X_processed = df
        
        # Predict
        probs = model.predict_proba(X_processed)
        fraud_prob = float(probs[0, 1]) # Probability of class 1 (Fraud)
        
        is_fraud = bool(fraud_prob >= FRAUD_THRESHOLD)
        logger.info(f"Prediction complete. Fraud Probability: {fraud_prob:.4f}, Is Fraud: {is_fraud}")
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            is_fraud=is_fraud
        )
    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest, artifacts: dict = Depends(get_prediction_artifacts)):
    """
    Predicts fraud probabilities for a batch of transactions.
    """
    try:
        batch_size = len(request.transactions)
        logger.info(f"Received batch prediction request for {batch_size} transactions.")
        
        # Convert requests to DataFrame
        df = pd.DataFrame([req.model_dump() for req in request.transactions])
        
        pipeline = artifacts.get("pipeline")
        model = artifacts.get("model")
        
        if pipeline is not None:
            X_processed = pipeline.transform(df)
        else:
            X_processed = df
        
        # Predict
        probs = model.predict_proba(X_processed)
        fraud_probs = probs[:, 1].astype(float)
        
        responses = [
            PredictionResponse(
                fraud_probability=prob,
                is_fraud=bool(prob >= FRAUD_THRESHOLD)
            ) for prob in fraud_probs
        ]
        
        logger.info(f"Successfully processed batch of {batch_size} transactions.")
        return BatchPredictionResponse(predictions=responses)
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
