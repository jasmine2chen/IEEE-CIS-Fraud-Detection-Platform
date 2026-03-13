from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from typing import Any, Dict

def get_xgboost_model(params: Dict[str, Any] = None) -> XGBClassifier:
    """Instantiate an XGBoost classifier with given params."""
    if params is None:
        params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
    return XGBClassifier(**params, eval_metric="logloss")
