import mlflow
import argparse
import pandas as pd
import numpy as np
from src.data_prep.data_loader import prepare_data
from src.features.build_features import build_features, get_feature_pipeline
from src.models.tree_models import get_xgboost_model
from sklearn.metrics import roc_auc_score

def time_consistency_split(df: pd.DataFrame, train_months: int = 1, test_month_offset: int = 5):
    """
    Split data based on Time Consistency rules (e.g. train on month 1, test on month 6).
    Assuming TransactionDT is in seconds and starts near 0.
    1 month ~ 30 days = 2,592,000 seconds.
    """
    SECONDS_IN_MONTH = 2592000
    df['Month'] = np.floor(df['TransactionDT'] / SECONDS_IN_MONTH)
    
    # Simple split for demonstration (e.g., train on first month, test on last available)
    train_idx = df[df['Month'] < train_months].index
    test_idx = df[df['Month'] == df['Month'].max()].index 
    
    return train_idx, test_idx

def train(trans_path: str, id_path: str):
    """Main training loop using Time Consistency Evaluation."""
    mlflow.set_experiment("fraud_detection_kaggle_magic")
    
    with mlflow.start_run():
        print("Loading and preparing data...")
        X, y = prepare_data(trans_path, id_path)
        
        print("Building features (UID, D-Norm)...")
        X = build_features(X)
        
        train_idx, test_idx = time_consistency_split(X)
        
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        cols_to_drop = ['TransactionDT', 'uid', 'Month']
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])
        
        print("Applying Preprocessing Pipeline (Imputing & Frequency Encoding)...")
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        pipeline = get_feature_pipeline(numeric_cols, categorical_cols)
        
        # Fit and transform the training data
        X_train_processed = pipeline.fit_transform(X_train)
        
        # Transform the testing data
        X_test_processed = pipeline.transform(X_test)
        
        print(f"Training XGBoost on {X_train_processed.shape[0]} samples with {X_train_processed.shape[1]} features...")
        model = get_xgboost_model()
        model.fit(X_train_processed, y_train)
        
        print("Evaluating on future month...")
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        
        print(f"Time-Consistency Out-of-Time AUC: {auc:.4f}")
        mlflow.log_metric("OOT_AUC", auc)
        
        # Determine if we should keep specific engineered features based on AUC delta
        # (This is where the iterative column dropping would happen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", type=str, required=True, help="Path to raw transaction data")
    parser.add_argument("--id", type=str, required=True, help="Path to raw identity data")
    args = parser.parse_args()
    
    train(args.trans, args.id)
