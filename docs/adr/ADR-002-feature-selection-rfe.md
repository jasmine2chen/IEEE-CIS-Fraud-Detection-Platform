# ADR-002: Stability-Weighted RFE for Feature Selection

**Status:** Accepted
**Date:** 2026-01-22
**Authors:** ML Research Team

---

## Context

The IEEE-CIS dataset contains 400+ raw features. In production, every feature carries compute cost (fetch, transform, validate) and latency cost at inference time. We need a feature selection strategy that:

1. Removes features that are both weak and temporally unstable
2. Does not inadvertently remove features with strong but erratic signal
3. Is compatible with both tree models (XGBoost) and neural hybrid models (`[orig ‖ embed]`)
4. Produces a deterministic, auditable feature set written to `model_config.yaml`

Standard recursive feature elimination (RFE) uses a single held-out validation score, which can select features that happen to fit recent patterns but are unreliable over time. For fraud, this is a critical failure mode: a feature that predicts fraud well in months 1–5 but degrades in month 6 will be selected despite being operationally unreliable.

---

## Decision

We use **Upper Confidence Bound (UCB) stability-weighted RFE**:

```
UCB score = mean_FPR_across_folds + k × std_FPR_across_folds
```

Where:
- `mean_FPR` is the mean FPR at the operating threshold across expanding-window CV folds
- `std_FPR` is the standard deviation — a proxy for temporal instability
- `k` controls the stability-vs-performance trade-off (default `k=1`)
- A **lower UCB score is better** (lower FPR, lower variance)

Features are eliminated iteratively by removing the least important feature (by XGBoost `gain` importance) and re-evaluating the UCB score until removing the next feature would increase the UCB score beyond a tolerance threshold.

---

## Rationale

### Why FPR, not AUC or log-loss?

AUC and log-loss are calibration metrics that reward the model for ranking all examples correctly. In production, we only care about performance within the operational FPR range (0–5%). Features that improve AUC in the tail of the score distribution (very high scores) but degrade precision at the operating threshold are not useful.

FPR at the operating threshold directly measures what the fraud ops team experiences: review queue pressure.

### Why UCB, not just mean FPR?

A feature set with mean FPR = 0.015 ± 0.008 across folds is a deployment risk: in a bad month it may hit 0.023 (false positive review queue 50% over capacity). A feature set with mean FPR = 0.018 ± 0.001 is more operationally reliable despite having worse mean performance.

The UCB criterion formalises this trade-off: it treats feature selection as a bandit problem where we want to select the arm (feature set) with the best worst-case performance under uncertainty.

### Why recency-weighted CV folds?

Standard k-fold CV weights all time periods equally. Fraud patterns shift over time — a model that fits months 1–3 well but degrades in months 4–6 will show inflated average CV performance. Recency weighting (weights `[1, 2, 3]` for 3 folds, most recent fold 3× weight) penalises temporal degradation during selection, aligning CV performance with OOT test performance.

### L1 vs L2 RFE

| Model Type | RFE Norm | Rationale |
|---|---|---|
| XGBoost | L1 (Lasso-style) | Original features are high-dimensional; we want sparse selection |
| Neural hybrid | L2 (Ridge-style) | `[orig ‖ embed]` matrix is dense; L2 distributes importance across correlated features rather than zeroing them |

For neural hybrids, embedding dimensions are correlated by construction. L1 would arbitrarily zero some dimensions while keeping others; L2 gracefully handles the collinearity.

---

## Consequences

**Positive:**
- Selected feature set is written to `model_config.yaml` — a human-readable, version-controlled artifact
- Reduces feature count from 400+ to ~50–80 features, cutting inference latency by ~40%
- Stability gate (final step: `var(fold FPRs) < 0.03`) acts as a circuit breaker before promotion

**Negative:**
- UCB RFE adds ~20% wall-clock time to the tuning pipeline compared to simple importance thresholding
- The UCB hyperparameter `k` requires domain judgment; we default to `k=1` (equal weight on mean and variance)
- Re-running RFE after data refresh can produce a different feature set, requiring pipeline redeployment

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Variance threshold | Removes features with low variance, not low signal — unsuitable for binary fraud where rare events have high variance |
| SHAP-based importance cutoff | SHAP importance computed post-hoc on a single model; no stability signal across folds |
| Mutual information filter | Non-parametric, computationally expensive at 400+ features; no temporal stability signal |
| No feature selection | 400+ features increases latency, risks overfitting, and makes model harder to audit for governance |
| Permutation importance | Correlated features distribute importance; permuting one feature doesn't fully remove its signal — biased toward retaining correlated groups |
