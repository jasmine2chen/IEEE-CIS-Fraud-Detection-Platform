# IEEE-CIS Fraud Detection Platform

![CI](https://img.shields.io/github/actions/workflow/status/jasmine2chen/IEEE-CIS-Fraud-Detection-Platform/ci.yml?label=tests&logo=github)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellowgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

A **production-grade ML platform** for credit card fraud detection covering the full ML lifecycle — deep learning & graph-based architecture benchmarking, Bayesian HPO with automated retraining, and a complete AWS MLOps stack. Built to reflect senior ML engineering standards: operability, temporal robustness, and maintainability at scale.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Model Architectures](#model-architectures)
- [HPO & Automated Retraining](#hpo--automated-retraining)
- [MLOps Stack](#mlops-stack)
- [Project Layout](#project-layout)
- [Evaluation](#evaluation)
- [Configuration](#configuration)

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Place Kaggle CSVs in data/raw/
unzip ieee-fraud-detection.zip -d data/raw/

# Tune → write best params to config → train
make tune-then-train MODEL=xgboost TRIALS=50
make tune-then-train MODEL=mlp_xgboost TRIALS=50
make tune-then-train MODEL=transformer_xgboost TRIALS=30
make tune-then-train MODEL=gnn_xgboost TRIALS=25

# Serve locally
make run-api                       # → http://localhost:8000/docs

# Full observability stack (API + MLflow UI + Grafana)
make stack-up

# Cross-model benchmark (all 4 architectures)
make benchmark
```

---

## Model Architectures

Four architectures trained on the same feature set and evaluated on an identical OOT test harness for controlled comparison. All hybrid models follow the same two-stage pattern: a neural encoder pre-trained with FocalLoss generates embeddings that are concatenated to the original features before XGBoost training.

| Model | Architecture | Stage 1 | Stage 2 |
|---|---|---|---|
| `xgboost` | Gradient-boosted trees | — | 9-param Bayesian HPO; FPR early stopping; L1 RFE |
| `mlp_xgboost` | MLP → XGBoost | FocalLoss MLP encoder; ReLU + BN + Dropout | XGBoost on `[orig ‖ mlp_embed]`; L2 RFE |
| `transformer_xgboost` | FT-Transformer → XGBoost | Per-feature linear projection → learnable positional encodings → TransformerEncoder (pre-LN); mean pool | XGBoost on `[orig ‖ transformer_embed]`; L2 RFE |
| `gnn_xgboost` | GraphSAGE → XGBoost | 2-layer SAGE with historical neighbour aggregation by `card1` identity; FocalLoss | XGBoost on `[orig ‖ gnn_embed]`; L2 RFE |

### FT-Transformer (TabTransformer)

Each scalar feature is independently projected into a shared `d_model`-dimensional space via `Linear(1, d_model)`. Learnable positional encodings are added per feature index. A stack of `TransformerEncoder` layers with pre-layer-norm (`norm_first=True`) and multi-head attention learns cross-feature interactions. The output token sequence is mean-pooled to a fixed-size embedding.

### GraphSAGE (Inductive GNN)

Transactions are linked by shared `card1` identity. A 2-layer SAGE network aggregates neighbour features via concatenation + linear transform + BN + ReLU. To avoid building an N×N adjacency matrix, training uses a *historical aggregation* pattern: `neigh_h0` (mean raw features per card) is static; `neigh_h1` (mean first-layer outputs per card) is refreshed after each epoch from the current encoder state.

Inference is fully inductive: `card_h0_mean.pkl` and `card_h1_mean.pkl` lookup tables are saved at training time so the API approximates SAGE aggregation without graph access. Unseen cards fall back to zero vectors.

```bash
make train MODEL=transformer_xgboost
make train MODEL=gnn_xgboost
```

---

## HPO & Automated Retraining

### Bayesian HPO Pipeline (Optuna TPE)

Each model type runs a structured multi-phase tuning pipeline in `src/training/tune.py`:

**XGBoost (4-step):**
1. Optuna TPE over 9 hyperparameters with recency-weighted temporal CV
2. Stability-gated L1 RFE — eliminates features only when `mean_FPR + k·std_FPR` improves (UCB gating prevents unstable feature removal)
3. Re-tune on selected feature set
4. Stability gate: `var(fold FPRs) < 0.03` must pass before writing params to config

**Neural hybrids (2-phase):**
- **Phase A** — Encoder HPO: Optuna over architecture + learning rate hyperparameters, evaluated on a single OOT split (CV is too expensive with GPU training)
- **Phase B** — XGBoost HPO: freeze best encoder, extract `[orig ‖ embed]` matrix, run stability-gated RFE + re-tune on embeddings

### S3 Config Writeback

After each tuning run, `tune.py` writes the updated `model_config.yaml` back to S3 (set `S3_CONFIG_BUCKET` env var to enable). Scheduled ECS retraining tasks pull the latest config at startup, enabling zero-code drift response without a code deploy.

```bash
export S3_CONFIG_BUCKET=my-fraud-model-configs
make tune MODEL=xgboost TRIALS=50   # → uploads configs/model_config.yaml to S3
```

### Recency-Weighted CV

Expanding-window temporal folds are weighted `[1, 2, 3]` (most recent fold counts 3×). This reflects that recent fraud patterns should dominate the HPO objective over older transactions.

---

## MLOps Stack

| Layer | Technology | Details |
|---|---|---|
| **Experiment tracking** | MLflow on EC2 | Params, metrics, artifacts, SHAP plots logged per run; `@champion`/`@challenger` model aliases |
| **Orchestration** | Prefect | `pipelines/training_pipeline.py` — tune → train → register → stability gate → promote |
| **Serving** | FastAPI on ECS | `/predict`, `/predict_batch`, `/health`; shadow mode A/B with `@challenger`; `asyncio.to_thread` for non-blocking inference |
| **Artifact storage** | S3 | Model checkpoints, HPO configs, feature pipelines |
| **Drift monitoring** | Evidently + Grafana | KS test on score distribution; Jensen-Shannon divergence per feature; 7-day rolling FPR |
| **CI/CD** | GitHub Actions | CI: lint + tests on every PR; CD: ECR push → ECS task-def render → rolling deploy on merge to `main` |

### CI/CD Pipeline

```
push to main
    │
    ├─ integration-test    pytest -m "not slow" + registry import check
    │
    ├─ build-and-push-ecr  docker build → tag with SHA → push to ECR
    │
    ├─ deploy-ecs          render new task-def (inject MODEL_TYPE, MLFLOW_TRACKING_URI,
    │                      S3_CONFIG_BUCKET) → ECS rolling deploy → wait for stability
    │
    └─ notify              Slack on success / failure
```

Required GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`, `ECS_TASK_DEFINITION`, `CONTAINER_NAME`, `MLFLOW_TRACKING_URI`, `S3_CONFIG_BUCKET`, `SLACK_WEBHOOK_URL` (optional).

### Champion / Challenger Registry

```python
from src.deployment.registry import promote_to_champion, load_champion

# After training a new version
promote_to_champion("transformer_xgboost", version="3")

# API loads @champion at startup; @challenger runs in shadow mode
pipeline, model = load_champion("transformer_xgboost")
```

---

## Project Layout

```
fraud_detection/
├── configs/
│   └── model_config.yaml          Hyperparameters for all 4 model types
├── pipelines/
│   └── training_pipeline.py       Prefect flow: tune→train→register→promote
├── src/
│   ├── training/
│   │   ├── models/
│   │   │   ├── mlp_tree.py        MLP encoder + FocalLoss + EarlyStopping
│   │   │   ├── transformer_tree.py  FT-Transformer encoder
│   │   │   └── gnn_tree.py        GraphSAGE encoder + lookup-table inference
│   │   ├── train.py               OOT split, MLflow tracking, all 4 model types
│   │   └── tune.py                Optuna HPO, RFE, S3 config writeback
│   ├── evaluation/
│   │   └── benchmark.py           FPR sweep + dollar recall, all 4 models
│   ├── deployment/
│   │   ├── registry.py            MLflow @champion/@challenger alias management
│   │   ├── api/main.py            FastAPI serving with shadow mode
│   │   └── batch_score.py         Offline batch scoring
│   └── monitoring/
│       └── drift.py               Evidently drift reports
├── .github/workflows/
│   ├── ci.yml                     Lint + tests on every PR
│   └── cd.yml                     ECR push + ECS deploy on merge to main
└── docker-compose.yml             fraud_api + MLflow UI + Grafana
```

---

## Evaluation

All models are evaluated on a strict temporal OOT split (month 6 held out). Business metrics prioritise dollar recall and FPR sweep over raw AUC.

| Model | ROC-AUC | PR-AUC | pAUC@5%FPR | Recall@2%FPR | $Recall@2%FPR |
|---|---|---|---|---|---|
| `xgboost` | ~0.925 | ~0.710 | ~0.875 | ~0.820 | ~0.860 |
| `mlp_xgboost` | ~0.930 | ~0.725 | ~0.882 | ~0.835 | ~0.870 |
| `transformer_xgboost` | ~0.933 | ~0.731 | ~0.886 | ~0.841 | ~0.876 |
| `gnn_xgboost` | ~0.935 | ~0.738 | ~0.889 | ~0.848 | ~0.883 |

*Indicative values — run `make benchmark` for exact figures on your data.*

The default operating threshold (`fraud_threshold_prob: 0.85`) targets a **2% FPR** to match typical fraud ops review queue capacity.

---

## Configuration

`configs/model_config.yaml` is the single source of truth for all hyperparameters. `tune.py` writes back the best found params after each run; S3 writeback propagates them to scheduled ECS retraining without a code deploy.

```yaml
model:
  type: "gnn_xgboost"  # xgboost | mlp_xgboost | transformer_xgboost | gnn_xgboost

gnn_xgboost_params:
  hidden_dim: 64
  out_dim: 32
  dropout_rate: 0.1
  learning_rate: 0.001
  encoder_epochs: 15
  batch_size: 2048
  max_neighbors: 30
  ...
```

See [docs/model_card.md](docs/model_card.md) for governance, fairness considerations, and SR 11-7 compliance notes.
