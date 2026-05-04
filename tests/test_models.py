import numpy as np
import pytest
import torch

from src.training.models.mlp_tree import EarlyStopping, FocalLoss, MLPEncoder, train_mlp_xgboost


def test_mlp_encoder_forward():
    """Forward pass produces correct output shape [N, last_hidden_dim]."""
    model = MLPEncoder(input_dim=10, hidden_dims=[32, 16], dropout_rate=0.1)
    x = torch.randn(4, 10)
    output = model(x)
    assert output.shape == (4, 16)


def test_focal_loss():
    """FocalLoss returns a positive scalar."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logits  = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([1.0, 1.0,  0.0])
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_train_mlp_xgboost_smoke():
    """Smoke test: full MLP+XGBoost pipeline runs and returns two models."""
    np.random.seed(42)
    torch.manual_seed(42)
    X_train = np.random.randn(200, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(200,)).astype(np.float32)
    X_val   = np.random.randn(40,  10).astype(np.float32)
    y_val   = np.random.randint(0, 2, size=(40,)).astype(np.float32)

    params = {
        "hidden_dims": [32, 16],
        "dropout_rate": 0.1,
        "learning_rate": 0.01,
        "encoder_epochs": 2,
        "batch_size": 32,
        "patience": 5,
        "n_estimators": 10,
        "max_depth": 3,
        "tree_method": "hist",
    }
    encoder, xgb_model = train_mlp_xgboost(
        X_train, y_train, X_val, y_val,
        params=params, save_path=None,
    )
    assert isinstance(encoder, MLPEncoder)
    preds = xgb_model.predict_proba(
        np.concatenate([X_val, encoder(torch.FloatTensor(X_val)).detach().numpy()], axis=1)
    )
    assert preds.shape == (40, 2)


def test_early_stopping():
    """EarlyStopping counter increments and triggers at patience."""
    es = EarlyStopping(patience=2, min_delta=0.01)
    model = torch.nn.Linear(10, 1)

    es(1.0, model)           # first call — sets best
    assert not es.early_stop
    assert es.counter == 0

    es(1.0, model)           # no improvement
    assert not es.early_stop
    assert es.counter == 1

    es(1.0, model)           # counter hits patience=2 → trigger
    assert es.early_stop


def test_early_stopping_resets_on_improvement():
    """Counter resets when the metric improves."""
    es = EarlyStopping(patience=3)
    model = torch.nn.Linear(10, 1)
    es(1.0, model)
    es(1.0, model)           # counter = 1
    es(0.5, model)           # improvement — counter resets
    assert es.counter == 0
    assert not es.early_stop
