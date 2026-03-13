import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_single_endpoint():
    """Test the /predict endpoint for single transactions."""
    payload = {
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 321.0,
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "TransactionDT": 86450,
        "D1": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "model_version" in data
    
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["is_fraud"], bool)

def test_predict_batch_endpoint():
    """Test the /predict_batch endpoint."""
    payload = {
        "transactions": [
            {
                "TransactionAmt": 150.0,
                "ProductCD": "W",
                "card1": 13926,
                "TransactionDT": 86450,
            },
            {
                "TransactionAmt": 25.5,
                "ProductCD": "C",
                "card1": 12345,
                "TransactionDT": 90000,
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    for pred in data["predictions"]:
        assert 0.0 <= pred["fraud_probability"] <= 1.0
        assert isinstance(pred["is_fraud"], bool)

def test_validation_error():
    """Test that the schema strictly rejects invalid types/missing fields."""
    # Missing required field `TransactionDT`
    payload = {
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 13926,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Standard FastAPI validation error
