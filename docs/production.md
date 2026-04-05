# Production Guide

This document answers three questions every ML team should be able to answer before shipping a model:

1. [How does this work in production?](#1-production-architecture--monitoring)
2. [What does it cost at scale?](#2-infrastructure-cost-analysis)
3. [What could go wrong, and how do we know?](#3-failure-modes--detection)

---

## 1. Production Architecture & Monitoring

### Deployment topology

```
                          ┌─────────────────────────────────┐
                          │  Kubernetes / ECS cluster        │
  Payment Gateway  ──────►│                                  │
                          │  fraud-api  (n replicas)         │
  Mobile App       ──────►│  ├── /predict       p50 < 20ms  │
                          │  ├── /predict_batch  p99 < 80ms  │
  Batch Scorer    ──────►│  └── /health                     │
                          │                                  │
                          │  Prometheus scrape (:8001/metrics)│
                          └─────────────┬───────────────────┘
                                        │
                          ┌─────────────▼───────────────────┐
                          │  Grafana dashboard               │
                          │  • Request rate & error rate     │
                          │  • p50/p95/p99 latency           │
                          │  • Fraud rate (rolling 1h/24h)   │
                          │  • Score distribution histogram  │
                          └─────────────────────────────────┘
```

### Key metrics to instrument

Every response from `/predict` should emit the following Prometheus metrics:

| Metric | Type | Labels | Alert threshold |
|---|---|---|---|
| `fraud_api_requests_total` | Counter | `endpoint`, `status_code` | error rate > 0.1% |
| `fraud_api_latency_seconds` | Histogram | `endpoint` | p99 > 100ms |
| `fraud_score_distribution` | Histogram | — | see drift section |
| `fraud_rate_1h` | Gauge | — | > 3× baseline 30-day avg |
| `model_version` | Gauge | `version` | version mismatch alert |

```python
# Example instrumentation (prometheus_client)
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT  = Counter("fraud_api_requests_total", "...", ["endpoint", "status_code"])
LATENCY        = Histogram("fraud_api_latency_seconds", "...", ["endpoint"],
                           buckets=[.005, .01, .025, .05, .1, .25, .5, 1.0])
SCORE_DIST     = Histogram("fraud_score_distribution", "...",
                           buckets=[0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, 1.0])
FRAUD_RATE_1H  = Gauge("fraud_rate_1h", "Rolling 1-hour fraud rate")
```

### Model monitoring (data drift)

Fraud patterns shift over time (new attack vectors, seasonal behaviour). A static model degrades silently. Run the following checks nightly:

| Check | Method | Alert if |
|---|---|---|
| **Score distribution drift** | KS-test: prod scores vs training holdout | p-value < 0.05 |
| **Feature drift** | Population Stability Index on top-20 features | PSI > 0.2 on any feature |
| **Label drift** | Compare fraud rate in confirmed labels vs 30-day baseline | > 2× or < 0.5× |
| **Concept drift** | Re-evaluate on last 30 days of confirmed labels (delayed feedback) | AUC@5%FPR drops > 3pp |

Trigger a retraining pipeline if two or more drift signals fire simultaneously.

### Alerting runbook

**P1 — Immediate page**
- Error rate > 1% for > 2 minutes
- p99 latency > 500ms for > 5 minutes
- Model artifacts missing (health endpoint reports `"model_loaded": false`)

**P2 — Next business hour**
- Fraud rate 3× above 30-day rolling average for > 1 hour
- KS-test score drift alert fires
- Daily AUC@5%FPR drops > 3pp vs last-week checkpoint

**P3 — Weekly review**
- PSI creeping above 0.1 on more than 3 features
- Average transaction amount shifting more than 1 std from baseline

---

## 2. Infrastructure Cost Analysis

### Assumptions: 1M predictions / day

| Parameter | Value |
|---|---|
| Daily volume | 1,000,000 transactions |
| Peak QPS | ~120 (assuming 80/20 distribution over business hours) |
| Model inference time | ~5ms per transaction (XGBoost, single core) |
| Feature pipeline time | ~3ms per transaction |
| Total per-request latency (p50) | ~10ms |

### Compute (AWS example, us-east-1, 2025 pricing)

| Component | Spec | Count | Monthly cost |
|---|---|---|---|
| **API servers** | t3.medium (2 vCPU, 4 GB) | 3 (HA, 1 spare) | ~$90 |
| **Load balancer** | ALB | 1 | ~$20 |
| **Model artefact storage** | S3 (< 1 GB) | 1 bucket | ~$0.02 |
| **MLflow tracking server** | t3.small | 1 | ~$15 |
| **Metrics / alerting** | Managed Prometheus + Grafana | — | ~$40 |
| **CI runners** | GitHub Actions (2,000 free min/month) | — | ~$0 |
| **Total** | | | **~$165 / month** |

Cost per prediction: **$0.000165** (< 0.02 cents)

### Scaling to 10M predictions / day

At 10× volume, peak QPS reaches ~1,200. The bottleneck shifts from compute to feature pipeline I/O.

| Component | Change | Additional cost |
|---|---|---|
| API servers | t3.medium × 3 → c6i.xlarge × 6 | +$350 |
| Redis feature cache | cache.t3.micro (hot uid lookups) | +$15 |
| RDS read replica | For audit log queries | +$50 |
| **Total at 10M/day** | | **~$580 / month** |

Cost per prediction: **$0.0000580** — unit cost drops 3× due to fixed overhead amortisation.

### Batch scoring path (offline use case)

For nightly batch scoring (e.g., reviewing prior-day transactions):

| Approach | Spec | Time for 1M records | Cost |
|---|---|---|---|
| Single EC2 | c6i.2xlarge | ~8 min | ~$0.03 per run |
| AWS Batch | Spot c6i.2xlarge | ~6 min (parallel) | ~$0.01 per run |
| EMR Spark | r6i.xlarge × 4 | ~4 min | ~$0.05 per run |

Recommendation: **AWS Batch on spot** for cost-optimised batch scoring.

---

## 3. Failure Modes & Detection

### Failure taxonomy

```
Failure
├── Infrastructure
│   ├── F1: Model artifacts missing / corrupted
│   ├── F2: Dependency version mismatch (sklearn pipeline incompatibility)
│   └── F3: Memory exhaustion (large batch requests)
├── Data Quality
│   ├── F4: Unexpected null pattern (upstream schema change)
│   ├── F5: Feature distribution shift (new card BIN ranges, new device types)
│   └── F6: Temporal leakage in online feature aggregation
└── Model Degradation
    ├── F7: Concept drift (new fraud attack vectors)
    ├── F8: Threshold miscalibration (fraud rate change shifts operating point)
    └── F9: Silent failure (model returns 0.0 for all inputs)
```

### F1 — Model artifacts missing

**Symptom:** `/health` returns `"model_loaded": false`; all `/predict` calls return 503.

**Detection:** Health check endpoint inspects artifact paths at startup. Liveness probe fails → Kubernetes restarts pod.

**Mitigation:**
- Store artefacts in versioned S3 with lifecycle policy (never delete, only archive)
- Pin model version in deployment config
- Blue/green deployment: old pods serve until new pods pass health check

---

### F2 — sklearn pipeline version mismatch

**Symptom:** `ValueError: n_features_in_` mismatch, or `AttributeError` on an unknown transformer attribute. Affects any deployment where the serving environment's sklearn version differs from training.

**Detection:** CI pipeline tests loading the serialised joblib artefact with the pinned sklearn version before promoting to staging.

**Mitigation:**
- Pin all dependency versions in `requirements.txt` (already done)
- Bake the exact Python environment into the Docker image layer
- Tag artefacts with the sklearn version used at fit time: `feature_pipeline_sklearn1.6.1.joblib`

---

### F3 — Memory exhaustion on large batch requests

**Symptom:** OOM kill or 502 from ALB when `/predict_batch` receives thousands of rows.

**Detection:** Container memory metric > 90% for > 30s.

**Mitigation:**
```python
# In api/schemas.py, add a max-length validator:
class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_length=500)
```
- Set hard limit: 500 transactions per batch call
- For larger volumes, use the async batch scoring path (S3 → Lambda → S3)

---

### F4 — Upstream schema change

**Symptom:** New column names or missing columns cause `KeyError` in the feature pipeline.

**Detection:**
- Request payload schema validation at the API layer (Pydantic — already in place)
- Column presence check in `build_features.py` with explicit `warnings.warn` for missing IEEE-CIS columns

**Mitigation:**
- Add schema version header to API requests: `X-Schema-Version: 1`
- Fail loudly on missing required features; impute gracefully on optional ones
- Contract tests in CI: load a golden request payload and assert column names match expected set

---

### F5 — Feature distribution shift

**Symptom:** Fraud rate appears normal but recall is dropping; model is consistently missing a new attack vector.

**Detection:**
- PSI > 0.2 on `card1`, `addr1`, `DeviceInfo` (high-cardinality entity features prone to new values)
- Frequency encoder will map unseen values to a low frequency — watch for spike in `freq_encoding = 0` counts

**Mitigation:**
- Retrain FrequencyEncoder monthly to absorb new card BIN ranges
- Shadow deployment: run new retrained model in parallel, compare score distributions before cutover

---

### F6 — Temporal leakage in online aggregation

**Symptom:** Offline metrics (AUC, recall) are significantly better than online metrics. Gap widens over time.

**Detection:**
- Compare offline AUC@5%FPR (from OOT holdout) against online AUC computed from confirmed labels (with 7-day label delay)
- Gap > 5pp is a strong leakage signal

**Mitigation:**
- All uid aggregations use `shift(1)` (see `_expanding_nunique_shifted`) — strictly look-back
- Online serving must re-compute aggregations from the transaction log up to T-1, never T
- Add an integration test: assert that no feature uses same-transaction information

---

### F7 — Concept drift (new fraud vectors)

**Symptom:** Fraud rate rises but model fraud probability distribution stays flat (model misses the new pattern entirely).

**Detection:**
- Monitor false negative rate on confirmed fraud labels (delayed feedback, ~7 days)
- Alert when FNR for confirmed frauds rises > 10pp above 30-day baseline

**Mitigation:**
- Automated retraining trigger: when two drift signals fire or FNR spike is confirmed
- Short-cycle retraining: keep last 3 months of data, weight recent weeks 2×
- Active learning queue: route low-confidence (0.4–0.6 probability) transactions to analyst review

---

### F8 — Threshold miscalibration

**Symptom:** Fraud review queue suddenly 3× larger (threshold too low) or near-empty (threshold too high) with no volume change.

**Detection:**
- Alert on `fraud_rate_1h > 3× baseline` or `fraud_rate_1h < 0.3× baseline`
- Plot score histogram: bimodal distribution shifting means recalibration needed

**Mitigation:**
- Threshold is a config value (`configs/model_config.yaml: serving.fraud_threshold_prob`)
- Monthly recalibration: find threshold that achieves target FPR on the most recent 30-day confirmed labels
- Platt scaling / isotonic regression on validation set to improve probability calibration

---

### F9 — Silent failure (model returns constant output)

**Symptom:** Model returns 0.0 for every prediction; no errors, no alerts, fraud goes undetected.

**Detection:**
- Monitor score distribution variance: alert if stddev of `fraud_probability` over a 1-hour window < 0.01
- Canary requests: inject synthetic high-risk transactions (known patterns from training set) and assert `fraud_probability > 0.9`

**Mitigation:**
- Canary health check: 5 synthetic transactions per minute, alert if any score < 0.85
- Shadow model: keep previous model version running; alert if score correlation drops below 0.8

---

## 4. Retraining Pipeline

```
Trigger (drift alert or schedule)
    │
    ▼
Fetch last N months of confirmed labels
    │
    ▼
make tune-then-train MODEL=xgboost TRIALS=50
    │
    ▼
Evaluate on latest OOT holdout
    │
    ├── AUC@5%FPR ≥ current prod model?
    │       │
    │       ├── YES → shadow deploy for 24h
    │       │            compare score distributions
    │       │            gate on FPR ≤ prod model FPR
    │       │            → promote to production
    │       │
    │       └── NO  → open investigation ticket
    │                   do NOT auto-promote
    │
    └── Log all metrics to MLflow run tagged "retraining_candidate"
```

Recommended schedule: monthly full retrain, weekly incremental evaluation.
