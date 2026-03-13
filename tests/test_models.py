import pytest
import torch
import numpy as np
import pandas as pd
from src.models.pytorch_mlp import FraudMLP, FocalLoss
from src.models.train_nn import train_nn, EarlyStopping

def test_fraud_mlp_forward():
    """Test the forward pass of the PyTorch MLP architecture."""
    input_dim = 10
    batch_size = 4
    model = FraudMLP(input_dim=input_dim, hidden_dim=32, dropout_rate=0.1)
    
    # Create dummy tensor
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Assert output shape is (batch_size, 1)
    assert output.shape == (batch_size, 1)

def test_focal_loss():
    """Test Focal Loss computation."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Dummy logits and targets
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([1.0, 1.0, 0.0])
    
    loss = criterion(logits, targets)
    
    # Loss should be a scalar tensor
    assert loss.dim() == 0
    assert loss.item() > 0

def test_train_nn_smoke():
    """Smoke test to ensure the training loop runs without errors."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Dummy data
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    X_val = np.random.randn(20, 10).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(20,)).astype(np.float32)
    
    # Train for 2 epochs
    model = train_nn(X_train, y_train, X_val, y_val, epochs=2, batch_size=32, lr=0.01)
    
    assert isinstance(model, FraudMLP)
    
def test_early_stopping():
    """Test early stopping logic."""
    es = EarlyStopping(patience=2, min_delta=0.01)
    model = torch.nn.Linear(10, 1)
    
    # Epoch 1: val_loss = 1.0
    es(1.0, model)
    assert not es.early_stop
    assert es.counter == 0
    
    # Epoch 2: val_loss = 1.0 (no improvement)
    es(1.0, model)
    assert not es.early_stop
    assert es.counter == 1
    
    # Epoch 3: val_loss = 1.0 (no improvement, should trigger stop)
    es(1.0, model)
    assert es.early_stop
