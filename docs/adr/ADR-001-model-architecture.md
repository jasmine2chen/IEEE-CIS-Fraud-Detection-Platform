# ADR-001: Neural Encoder + XGBoost Hybrid Architecture

**Status:** Accepted
**Date:** 2026-01-15
**Authors:** ML Research Team

---

## Context

The fraud detection task is a highly imbalanced binary classification problem (~3.5% positive rate) on structured tabular data with 400+ raw features. We evaluated two candidate architectures:

1. **Pure XGBoost** — gradient-boosted trees on engineered features
2. **MLP → XGBoost** — dense encoder pre-trained with FocalLoss, embeddings concatenated to original features

The core architectural tension is between **model expressiveness** (neural encoders capture high-order interactions) and **operational requirements** (SHAP explainability for SR 11-7 model risk governance, inference latency, maintenance cost).

---

## Decision

We adopt a **hybrid architecture** for MLP+XGBoost: a neural encoder as a learned feature extractor in Stage 1, followed by a frozen-weight export of embeddings that are concatenated to original features (`[orig ‖ embed]`), which are then consumed by an XGBoost classifier in Stage 2.

Specifically:
- **XGBoost** remains the production-ready baseline with the lowest operational cost
- **MLP → XGBoost** targets use cases where high-order interaction features provide meaningful lift

---

## Rationale

### Why XGBoost as the final classifier (not end-to-end neural)?

| Requirement | XGBoost head | End-to-end neural |
|---|---|---|
| SHAP explainability | Native TreeExplainer | Requires post-hoc approximation |
| Missing value handling | Built-in | Requires imputation layer |
| Inference latency | <2ms p99 | 5–30ms p99 depending on depth |
| Regulatory auditability | Decision tree paths readable | Black box |
| Calibration | Well-calibrated with Platt | Requires separate calibration step |

The XGBoost head gives us a SHAP-compatible interface regardless of encoder complexity. This is a non-negotiable requirement for SR 11-7 model risk governance in production banking environments.

### Why FocalLoss for encoder pre-training?

Standard BCE at 3.5% fraud rate causes the encoder to collapse to a constant representation (predicts majority class). FocalLoss down-weights easy negatives, forcing the encoder to allocate representational capacity to the minority class. This is Stage 1 only — Stage 2 XGBoost uses `scale_pos_weight` instead.

### Why `[orig ‖ embed]` concatenation?

Ablation studies show that discarding original features in favour of embeddings alone degrades performance by ~2–4 AUC points. The encoder learns complementary representations; original features retain independent signal not captured by the encoder's inductive bias.

---

## Consequences

**Positive:**
- Both architectures share a single evaluation harness, enabling controlled comparison on identical OOT test splits
- XGBoost head is fully SHAP-compatible; model outputs are auditable to the feature level
- MLP encoder can be retrained independently of the XGBoost classifier (Stage A / Stage B decoupling in tuning)

**Negative:**
- Hybrid architecture requires a two-phase training pipeline with a checkpoint between stages
- Embedding dimensions must be fixed at training time; encoder architecture changes require full pipeline retraining

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| LightGBM instead of XGBoost | Marginal performance difference; XGBoost has better MLflow native logging and `best_iteration` support |
| CatBoost | Categorical handling is already handled by `FrequencyEncoder` in our feature pipeline; adds dependency without benefit |
| End-to-end TabNet | No native SHAP support; inference latency 10–15ms; not compatible with RFE-based feature selection |
| Random Forest baseline | ~3 AUC points below XGBoost; not competitive on partial AUC@5% FPR |
