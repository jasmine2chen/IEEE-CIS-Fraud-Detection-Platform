from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional


class TransactionRequest(BaseModel):
    """Schema for a single transaction prediction request.

    Required fields are the minimum set needed for feature engineering
    (UID construction, D-column normalisation). All remaining IEEE-CIS
    features (V-columns, C-columns, M-columns, id_ columns, etc.) are
    accepted transparently via `extra='allow'` and forwarded as-is to the
    pipeline, matching the full feature space the model was trained on.

    Callers should include as many features as available — the pipeline's
    SimpleImputer handles any that are missing.
    """

    # Core required fields
    TransactionAmt: float = Field(..., description="Transaction amount in USD")
    TransactionDT: int = Field(..., description="Seconds elapsed from dataset reference point")
    card1: int = Field(..., description="Card feature 1 (anonymised)")

    # Optional but important for feature engineering
    ProductCD: Optional[str] = Field(None, description="Product code")
    card2: Optional[float] = Field(None, description="Card feature 2")
    addr1: Optional[float] = Field(None, description="Billing region code")
    addr2: Optional[float] = Field(None, description="Billing country code")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    D1: Optional[float] = Field(None, description="Days since account creation (relative)")

    model_config = ConfigDict(
        # Accept all additional IEEE-CIS fields (V1-V339, C1-C14, M1-M9,
        # id_01-id_38, etc.) without enumerating them in the schema.
        extra="allow",
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
                "D1": 0.0,
                "V1": 1.0,
                "C1": 1.0,
                "M1": "T",
            }
        },
    )


class BatchTransactionRequest(BaseModel):
    """Batch prediction request — list of TransactionRequest objects."""
    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):
    """Prediction result for a single transaction."""
    transaction_id: Optional[str] = Field(None, description="Optional caller-supplied ID for tracking")
    fraud_probability: float = Field(..., description="Probability [0.0, 1.0] of fraud")
    is_fraud: bool = Field(..., description="True if fraud_probability exceeds the operating threshold")
    model_version: str = Field(default="v1.0", description="Model version that produced this prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
