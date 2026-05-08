# Model Card: IEEE-CIS Fraud Detection Platform

**Version:** 1.0
**Last Updated:** 2026-01-30
**Team:** ML Research Engineering

---

## Model Overview

| Field | Details |
|---|---|
| **Task** | Binary classification — transaction fraud detection |
| **Architectures** | XGBoost, MLP→XGBoost, FT-Transformer→XGBoost, GraphSAGE→XGBoost |
| **Primary metric** | Recall at 2% FPR (production operating point) |
| **Governance framework** | SR 11-7 model risk management |
| **Explainability** | SHAP TreeExplainer (all architectures via XGBoost head) |

---

## Intended Use

**Primary use case:** Real-time and batch fraud scoring for credit card transactions. The model outputs a fraud probability (`fraud_probability`) and binary decision (`is_fraud`) given transaction and identity features.

**Intended users:** Fraud operations teams, ML engineers, and model risk management reviewers.

**Out-of-scope uses:**
- Credit underwriting or credit risk scoring (model is not calibrated for that outcome)
- Identity verification
- Any use case where SR 11-7 explainability requirements cannot be met (do not use neural encoder outputs directly — always use the XGBoost head for scored decisions)

---

## Training Data

| Field | Details |
|---|---|
| **Dataset** | [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) |
| **Source** | Vesta Corporation — real-world e-commerce transactions |
| **Size** | ~590K transactions, ~400 features (transaction + identity tables) |
| **Fraud rate** | ~3.5% |
| **Time range** | ~6 months of transaction history |
| **Train / Test split** | Months 0–5 train, Month 6 OOT test (strict temporal split — no random shuffling) |

### Data Preprocessing

- Memory-optimised load: int64 → int32, float64 → float32 where safe
- `uid` composite key (`card1_addr1_P_emaildomain`) links transactions to cardholder identity
- D-column normalisation removes calendar drift from delta-day features
- `FrequencyEncoder` fitted on train split only (no leakage)

---

## Evaluation

All metrics computed on the held-out OOT test set (month 6 only). The OOT split is reproduced identically in `src/benchmark.py` for cross-model comparison.

### Benchmark Results (Reference)

| Model | ROC-AUC | PR-AUC | pAUC@5%FPR | Recall@2%FPR | $Recall@2%FPR | Brier↓ |
|---|---|---|---|---|---|---|
| `xgboost` | ~0.925 | ~0.710 | ~0.875 | ~0.820 | ~0.860 | ~0.025 |
| `mlp_xgboost` | ~0.930 | ~0.725 | ~0.882 | ~0.835 | ~0.870 | ~0.023 |
| `transformer_xgboost` | ~0.933 | ~0.731 | ~0.886 | ~0.841 | ~0.876 | ~0.022 |
| `gnn_xgboost` | ~0.935 | ~0.738 | ~0.889 | ~0.848 | ~0.883 | ~0.021 |

*Indicative values — run `make benchmark` for exact figures on your data version.*

### Primary Operating Point

The default operating threshold (`fraud_threshold_prob: 0.85` in `model_config.yaml`) is calibrated to a **2% FPR**, matching typical fraud ops review queue capacity. Adjust the threshold for different FPR targets.

---

## Limitations and Risks

### Known Limitations

**Temporal drift.** Fraud patterns evolve with new attack vectors (account takeover, synthetic identity, card testing). Models should be retrained quarterly or when drift monitoring triggers an alert. The stability gate (`var(fold FPRs) < 0.03`) and OOT evaluation mitigate — but do not eliminate — temporal degradation.

**Dataset geography.** IEEE-CIS data reflects Vesta Corporation's US e-commerce customer base. Performance on cross-border transactions, different merchant categories, or non-US card networks may differ.

**Class imbalance.** At 3.5% fraud rate, predicted probabilities near the decision boundary (0.40–0.70) have high uncertainty. Dollar recall is a more meaningful metric than transaction recall for large-amount edge cases.

### Bias and Fairness Considerations

The model does not use demographic features (age, race, gender, ZIP code) as inputs. However, transaction amount, card type, and email domain may correlate with demographic attributes in ways that are not fully characterised. Periodic fairness audits are recommended for production deployment.

---

## Explainability

All four architectures produce SHAP explanations via the XGBoost head:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_proc)
```

Feature importance is logged per training run via `log_feature_importance()` in `src/research_tools/experiment_logger.py`. Top-30 features by SHAP gain are stored as a CSV artifact in each MLflow run.

---

## Monitoring

| Signal | Method | Alert Threshold |
|---|---|---|
| Score distribution drift | KS test on `fraud_probability` | p < 0.05 |
| Feature distribution drift | Jensen-Shannon divergence (Evidently) | JS > 0.1 per feature |
| FPR at operating threshold | Rolling 7-day FPR | > 3.0% (1.5× target) |
| Prediction volume | Row count vs. expected | ± 20% |

See `src/monitoring/drift.py` and the Grafana dashboard at `http://localhost:3000` after `make stack-up`.

---

## Versioning and Reproducibility

- Every training run captures: git commit hash, branch, dirty status, package versions, full config, OOT evaluation metrics (MLflow)
- Model artifacts are registered in the MLflow Model Registry under the `@champion` alias
- The `research_run()` context manager in `src/research_tools/experiment_logger.py` enforces reproducibility at the experiment level
- `REQUIRE_CLEAN_GIT=1` blocks training from a dirty working tree

---

## Contact

For model risk governance, adversarial testing requests, or retraining decisions, open an issue in the project repository or contact the ML Research Engineering team.
