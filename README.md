# IEEE-CIS Fraud Detection Platform

![CI](https://img.shields.io/github/actions/workflow/status/jasminechen/fraud-detection/ci.yml?label=tests&logo=github)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen)
![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

A **production-grade ML platform** for credit card fraud detection covering the full ML lifecycle — feature engineering, model training, hyperparameter optimisation, offline evaluation, real-time serving, and monitoring. Designed to reflect senior ML engineering standards: operability, temporal robustness, and maintainability at scale, not just model accuracy.

---


## Table of Contents

- [Quickstart](#quickstart)
- [ML Lifecycle](#ml-lifecycle)
- [Architecture](#architecture)
- [Model Architecture](#model-architecture)

---

## Quickstart

```bash
# Install
pip install -e .

# Place Kaggle CSVs in data/raw/
unzip ieee-fraud-detection.zip -d data/raw/

# End-to-end: tune → write config → retrain
make tune-then-train MODEL=xgboost TRIALS=50

# Serve
make run-api                    # → http://localhost:8000/docs

# Full stack (API + MLflow UI)
make docker-build && make docker-run
```

---



---

## Architecture

```
data/raw/                       IEEE-CIS CSVs (transaction + identity)
    │
    ▼
src/data_prep/data_loader.py    Memory-optimised load + merge
    │
    ▼
src/features/build_features.py  Feature engineering + sklearn Pipeline
    │
    ▼
src/tune.py ───────────────────  Optuna HPO + RFE
    │                            ├── XGBoost: 4-step (HPO→RFE→re-HPO→gate)
    │                            └── Neural:  2-phase (encoder HPO → frozen
    │                                         embeddings → XGBoost RFE+HPO)
    ▼
src/train.py ──────────────────  OOT split | MLflow tracking
    ├── XGBoost                  Baseline — 9-param HPO, FPR early stopping
    └── MLP → XGBoost            FocalLoss encoder → [orig‖embed] → GBDT
    │
    ▼
api/main.py                     FastAPI — /predict  /predict_batch  /health
    │
    ▼
docker-compose.yml              fraud_api (port 8000) + MLflow UI (port 5000)
```

See [docs/architecture.md](docs/architecture.md) for full system diagrams.

---

## Model Architecture

Two architectures on the same feature set and evaluation harness for controlled comparison.

| Model | Architecture | Key Design Decisions |
|---|---|---|
| `xgboost` | Gradient-boosted trees | 9-param Bayesian HPO; FPR early stopping; L1 RFE on original features |
| `mlp_xgboost` | MLP encoder → XGBoost | FocalLoss pre-training; embeddings concat to original features; L2 RFE on `[orig‖embed]` |

The MLP encoder captures high-order feature interactions that axis-aligned tree splits miss. The XGBoost classifier handles missing values natively, provides fast inference, and keeps the model fully SHAP-explainable — a regulatory requirement in banking.

```bash
make train MODEL=xgboost
make train MODEL=mlp_xgboost
```



