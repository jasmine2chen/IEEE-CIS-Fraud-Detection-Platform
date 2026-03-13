# IEEE-CIS Fraud Detection Framework

A robust, production-grade Machine Learning pipeline for predicting high-dimensional, temporally-sensitive credit card fraud vectors, optimized for extremely imbalanced targets.

## Quickstart
1. **Unzip Data:** Extract Kaggle CSV files directly into the `data/raw/` directory.
2. **Setup:** Install the dependencies via `pip install -r requirements.txt`.
3. **Train:** Run `python src/train.py --trans path/to/trans --id path/to/idn`
4. **Deploy:** Start the local server `uvicorn api.main:app --reload` or use docker `docker build -t fraud_api . && docker run -p 8000:8000 fraud_api`

## Architecture Highlights
* **Memory Reduction:** Implements automated NumPy float downcasting to aggressively protect RAM usage.
* **Feature Engineering:** Drops generic One Hot Encoders for Memory-Safe `FrequencyEncoders`. Calculates temporal user-id logic using "Magic" UID tracking algorithms extracted specifically from the top Kaggle submissions.
* **Model Topologies:**
  1. XGBoost Pipeline optimized for high-sparsity trees.
  2. PyTorch `FraudMLP` Pipeline integrated with Focal Loss and Early Stopping.
* **Evaluation:** Deprecated K-fold CV for real-world Time-Consistency Splitting. Supports metric calculations strictly aimed at imbalance domains: ROC-AUC, PR-AUC, and continuous Metric Tracking logic.
