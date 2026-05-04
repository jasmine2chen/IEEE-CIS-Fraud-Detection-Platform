# IEEE-CIS Fraud Detection Platform

![CI](https://img.shields.io/github/actions/workflow/status/jasminechen/fraud-detection/ci.yml?label=tests&logo=github)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen)
![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

A **production-grade ML platform** for credit card fraud detection covering the full ML lifecycle — feature engineering, model training, hyperparameter optimisation, offline evaluation, real-time serving, and monitoring. Designed to reflect senior ML engineering standards: operability, temporal robustness, and maintainability at scale, not just model accuracy.

---

## Challenges

**Class imbalance.** Fraud rates are 0.1–3%. All models use FocalLoss for neural encoder pre-training and are evaluated at business-relevant FPR operating points, not AUC at a 0.5 threshold.

**Temporal distribution shift.** Fraud patterns drift with new attack vectors and seasonal spikes. The platform enforces strict out-of-time (OOT) evaluation (months 0–5 train, month 6 test) and recency-weighted cross-validation that penalises models which overfit to historical patterns.

**Operational feature cost.** In real-time scoring, every feature carries compute and latency cost. The tuning pipeline performs stability-weighted RFE — eliminating features that are both weak and temporally inconsistent — to minimise the deployed model's operational footprint.

---

## Table of Contents

- [Quickstart](#quickstart)
- [ML Lifecycle](#ml-lifecycle)
- [Architecture](#architecture)
- [Model Architecture](#model-architecture)
- [Feature Pipeline](#feature-pipeline)
- [Tuning Framework](#tuning-framework)
- [Evaluation Framework](#evaluation-framework)
- [Serving & Deployment](#serving--deployment)
- [MLOps](#mlops)
  - [Model Registry (MLflow)](#model-registry-mlflow)
  - [Orchestration (Prefect)](#orchestration-prefect)
  - [Batch Scoring (Offline)](#batch-scoring-offline)
  - [Drift Monitoring (Evidently + Grafana)](#drift-monitoring-evidently--grafana)
  - [CI/CD](#cicd)
- [Development](#development)
- [Repository Layout](#repository-layout)

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

## ML Lifecycle

```
1. Raw Data       IEEE-CIS transaction + identity CSVs
      ↓
2. EDA            notebooks/01_eda_and_baselines.ipynb
      ↓           Side-by-side model comparison, FPR sweep plots,
                  SHAP analysis, dollar recall curves
      ↓
3. Features       Magic UID, D-column normalisation, uid velocity
      ↓           aggregations, FrequencyEncoder, sklearn Pipeline
      ↓
4. Tuning         Optuna Bayesian HPO + RFE
      ↓           4-step pipeline (XGBoost) / 2-phase hybrid (neural)
                  Results written back to model_config.yaml
      ↓
5. Training       OOT split, FPR early stopping, MLflow tracking
      ↓
6. Offline Eval   FPR sweep, partial AUC@5% FPR, dollar recall, SHAP
      ↓
7. Serving        FastAPI real-time scoring, batch endpoint
      ↓
8. Monitoring     FPR drift alerting, score distribution tracking,
                  data quality checks
```

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

---

## Feature Pipeline

The pipeline is fitted **once** before any HPO trials and serialised to `models/feature_pipeline.joblib` — shared across all model types for consistent input representation and zero redundant computation.

| Feature | Description |
|---|---|
| `uid` | Magic UID composite key (`card1_addr1_P_emaildomain`) linking transactions to the same cardholder; enables velocity aggregations |
| D-column normalisation | Removes calendar drift from `D1`–`D15` delta-day columns |
| `uid` velocity aggregations | Transaction count and amount statistics per uid over rolling windows |
| `FrequencyEncoder` | Leakage-free categorical encoding; fitted on train split only |

---

## Tuning Framework

HPO and feature selection are designed around production constraints: high training cost, large data volume, and real-time inference latency budgets.

### XGBoost — Four-Step Pipeline

| Step | Method | Purpose |
|---|---|---|
| 1. Initial HPO | Optuna TPE, expanding-window CV, recency-weighted mean FPR | Weights [1,2,3] give the most recent fold 3× weight — incentivises capturing current fraud patterns |
| 2. RFE | UCB stability criterion: `mean_FPR + k×std_FPR` | Rewards feature sets that are *consistently* good across folds; directly reduces real-time feature compute cost |
| 3. Re-tune HPO | Bayesian HPO on selected features only | Optimal regularisation and tree depth shift as feature count drops; prevents underfitting |
| 4. Stability gate | `var(fold FPRs) < 0.03` | Flags models with high temporal variance before promotion; logged to MLflow |

### MLP+XGBoost — Two-Phase Pipeline

Multi-fold CV on neural encoder training is cost-prohibitive at production scale. The pipeline decouples encoder cost from XGBoost cost:

| Phase | Scope | Method |
|---|---|---|
| A — Encoder HPO | Encoder architecture only | Bayesian HPO, single OOT evaluation; fixed XGBoost base params as scoring proxy |
| A3 — Final training | Full dev set | Best encoder retrained once, then frozen permanently |
| B — XGBoost optimisation | `[original‖embeddings]` matrix | L2 RFE → re-HPO → stability gate on XGBoost only; cost is O(XGBoost) regardless of encoder complexity |

**L2 RFE** operates on the combined matrix — removing original features made redundant by encoder representations, and pruning uninformative embedding dimensions. All outputs are written to `model_config.yaml` as the single source of truth; `train.py` picks up changes with no code modification.

```yaml
# Populated by tune.py — consumed by train.py automatically
xgboost_selected_features: [0, 3, 7, ...]
mlp_xgboost_stage2_params: {n_estimators: 420, ...}
mlp_xgboost_selected: [0, 2, 5, 67, 68, ...]
```



---

## Evaluation Framework

Standard AUC at 0.5 threshold is not operationally meaningful for fraud — review queue capacity caps the false positive rate at 1–5%. All models are evaluated at business-relevant operating points:

| Metric | Definition |
|---|---|
| **FPR sweep** | Recall at every FPR point from 0.1% to 5% — mirrors how a fraud ops team sets a deployment threshold |
| **Partial AUC@5% FPR** | Model quality within the operational range only; normalised to [0,1] |
| **Dollar recall** | Fraction of total fraud *dollars* caught at each FPR — a £10K miss is not equivalent to a £10 miss |
| **FPR early stopping** | XGBoost optimises directly against the production FPR constraint during training, not log-loss |

All metrics are logged per run to MLflow for cross-model comparison.

---

## Serving & Deployment

### FastAPI Inference Service

```bash
# Single prediction
POST /predict
{"TransactionAmt": 150.0, "TransactionDT": 86400, "card1": 9500, ...}
→ {"fraud_probability": 0.03, "is_fraud": false, "model_version": "1.0.0"}

# Batch scoring
POST /predict_batch
{"transactions": [{...}, {...}]}

# Health check
GET /health
```

- Model and feature pipeline loaded once at startup via FastAPI lifespan — zero per-request cold start
- Pydantic v2 request/response validation with field-level error messages
- Feature pipeline and model artifact co-versioned to prevent schema drift between training and serving

### Docker

```bash
make docker-build   # multi-stage build — runtime image ~400MB
make docker-run     # fraud_api:8000 + mlflow:5000
```

---

## MLOps

### Experiment Tracking

Every run logs hyperparameters, per-fold FPR, FPR sweep metrics, dollar recall, model artifacts, and tags (`stability_gate_passed`, `tuning_framework`, `best_trial`) to MLflow.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db   # → http://localhost:5000
```

### Model Registry (MLflow)

Every trained model is automatically registered in the MLflow Model Registry using the `@champion` alias convention (MLflow 2.x — stage transitions are deprecated).

```python
# Aliases replace deprecated stage transitions
@champion   → model currently serving production traffic
@challenger → candidate model under A/B evaluation
```

```bash
make train MODEL=xgboost       # trains + auto-registers
make promote MODEL=xgboost     # promotes latest version to @champion
```

The API loads the `@champion` model at startup via `MLFLOW_TRACKING_URI`. Falls back to disk if the registry is unavailable.

### Orchestration (Prefect)

`pipelines/training_pipeline.py` defines a Prefect flow that chains tune → train → register → evaluate → promote as discrete tasks with dependency tracking, retries, and structured logging.

```bash
make pipeline-run MODEL=xgboost TRIALS=50   # local run, no server required
# or deploy to Prefect Cloud:
prefect deploy pipelines/training_pipeline.py:training_pipeline
```

The pipeline respects the stability gate: promotion to `@champion` is skipped if `var(fold FPRs) >= 0.03`, with a logged warning.

### Batch Scoring (Offline)

```bash
make batch-score MODEL=xgboost OUTPUT=data/predictions/batch_20260330.parquet
```

Loads `@champion` from the registry, scores a full transaction file, and writes a Parquet file with columns: TransactionID, fraud_probability, is_fraud, score_timestamp, model_version. Each run is logged to MLflow with n_scored, mean_fraud_prob, fraud_rate.

### Drift Monitoring (Evidently + Grafana)

```bash
make monitor MODEL=xgboost OUTPUT=data/predictions/batch_20260330.parquet
```

Compares the current scoring window against the training reference distribution using Evidently. Computes dataset drift (Jensen-Shannon divergence per feature) and prediction drift (KS test on score distribution). Results are persisted to PostgreSQL and visualised in Grafana.

```bash
make stack-up
# Grafana: http://localhost:3000  (admin/admin)
# Pre-built "Fraud Detection — Model Monitoring" dashboard
```

### CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | PR / push to main | pytest matrix (3.9, 3.11), mypy |
| `cd.yml` | Push to main | Integration tests → Docker build → push to registry → deployment notification |

### Temporal Stability & Drift Detection

Drift resistance is built into the training and tuning framework rather than added as an afterthought:

- **Recency-weighted CV** — models that only fit historical patterns score poorly at selection time
- **Stability gate** — `var(fold FPRs) < 0.03` flags distributional instability before production promotion
- **OOT evaluation** — strict temporal split prevents future leakage into feature statistics or model selection
- **FPR monitoring** — score distribution drift is visible as deviation from the expected operating FPR (`fraud_threshold_prob: 0.85`)

### Retraining Pipeline

```bash
make tune-then-train MODEL=xgboost      # tune → write config → retrain
make tune-then-train MODEL=mlp_xgboost
```

`tune-then-train` is designed to run as a scheduled job. Config writeback is the handoff between tuning and training — no code changes required to deploy a retrained model.

See [docs/production.md](docs/production.md) for Prometheus metric definitions, automated retraining trigger conditions, infrastructure cost analysis at 1M predictions/day, and failure mode playbooks.

---

## Development

```bash
pip install -e .
pip install pre-commit && pre-commit install
make test                    # pytest tests/ -v
pre-commit run --all-files   # black + ruff + isort + mypy
```


