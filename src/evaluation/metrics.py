import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss
)

def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes a comprehensive set of standard ML evaluation metrics for binary classification.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def print_evaluation_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    Prints a formatted, comprehensive classification report.
    """
    y_pred = (y_prob >= threshold).astype(int)
    print("=== Comprehensive ML Evaluation Report ===")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
    
    metrics = evaluate_classification(y_true, y_prob, threshold)
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"Brier Score:   {metrics['brier_score']:.4f}")
    print("-" * 42)
    print("Confusion Matrix:")
    print(f"TN: {metrics['confusion_matrix'][0][0]} | FP: {metrics['confusion_matrix'][0][1]}")
    print(f"FN: {metrics['confusion_matrix'][1][0]} | TP: {metrics['confusion_matrix'][1][1]}")
    print("==========================================")
