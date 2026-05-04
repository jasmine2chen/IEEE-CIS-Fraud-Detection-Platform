# Model Benchmark Results

| Model | ROC-AUC | PR-AUC | pAUC@5%FPR | Recall@1%FPR | Recall@2%FPR | Recall@5%FPR | $Recall@2%FPR | Brier↓ | Latency p50(ms) | Latency p99(ms) | N Trees |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.8307 | 0.4229 | 0.4248 | 0.3097 | 0.4218 | 0.5192 | 0.3602 | **0.0467** | **15.44** | **16.02** | **0** |
| mlp_xgboost | **0.8957** | **0.5118** | **0.5172** | **0.4277** | **0.5162** | **0.6342** | **0.4251** | 0.0486 | 15.59 | 16.94 | 0 |

*↓ = lower is better. Best value per column **bolded**. All models evaluated on the same OOT test split.*
