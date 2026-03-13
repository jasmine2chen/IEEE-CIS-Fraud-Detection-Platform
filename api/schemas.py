from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class TransactionRequest(BaseModel):
    """
    Schema for incoming transaction predictions.
    Fields represent a subset of the critical Kaggle features needed for the models.
    """
    TransactionAmt: float = Field(..., description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code for the transaction")
    card1: int = Field(..., description="Categorical card feature 1")
    card2: Optional[float] = Field(None, description="Categorical card feature 2")
    addr1: Optional[float] = Field(None, description="Billing zip code")
    addr2: Optional[float] = Field(None, description="Billing country code")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    TransactionDT: int = Field(..., description="Transaction timestamp delta from an arbitrary start date")
    D1: Optional[float] = Field(None, description="Time delta 1 (e.g. days since account creation)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TransactionAmt": 68.5,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 321.0,
                "addr1": 315.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "TransactionDT": 86400,
                "D1": 0.0
            }
        }
    )

class BatchTransactionRequest(BaseModel):
    """Batch prediction schema."""
    transactions: List[TransactionRequest]

class PredictionResponse(BaseModel):
    """
    Standard response schema for model predictions.
    Returns the probability of fraud and a boolean flag based on a threshold.
    """
    transaction_id: Optional[str] = Field(None, description="Optional ID for tracking")
    fraud_probability: float = Field(..., description="Probability [0-1] that the transaction is fraudulent")
    is_fraud: bool = Field(..., description="Boolean flag if probability exceeds operating threshold")
    model_version: str = Field(default="v1.0", description="Version of the model that served the prediction")
    
class BatchPredictionResponse(BaseModel):
    """Batch response schema."""
    predictions: List[PredictionResponse]
