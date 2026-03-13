from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from typing import List

from api.schemas import TransactionRequest, BatchTransactionRequest, PredictionResponse, BatchPredictionResponse

# Initialize FastAPI App
app = FastAPI(
    title="Fraud Detection API",
    description="Production API for serving real-time IEEE-CIS fraud detection predictions.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock model loading for Phase 4 scaffold
# In a real environment, this would load XGBoost or PyTorch weights from S3/MLflow
class MockModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Returns a random array of probabilities for demonstration
        return np.random.uniform(0, 1, size=(len(X), 2))

mock_model = MockModel()
FRAUD_THRESHOLD = 0.85 # High precision threshold setting

def get_model():
    """Dependency injector for model."""
    return mock_model

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: TransactionRequest, model: MockModel = Depends(get_model)):
    """
    Predicts fraud probability for a single transaction.
    """
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.model_dump()])
        
        # In a real deployment, we would apply `get_feature_pipeline` and `build_features` here
        
        # Predict
        probs = model.predict_proba(df)
        fraud_prob = float(probs[0, 1]) # Probability of class 1 (Fraud)
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            is_fraud=fraud_prob >= FRAUD_THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest, model: MockModel = Depends(get_model)):
    """
    Predicts fraud probabilities for a batch of transactions.
    """
    try:
        # Convert requests to DataFrame
        df = pd.DataFrame([req.model_dump() for req in request.transactions])
        
        # Predict
        probs = model.predict_proba(df)
        fraud_probs = probs[:, 1].astype(float)
        
        responses = [
            PredictionResponse(
                fraud_probability=prob,
                is_fraud=bool(prob >= FRAUD_THRESHOLD)
            ) for prob in fraud_probs
        ]
        return BatchPredictionResponse(predictions=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
