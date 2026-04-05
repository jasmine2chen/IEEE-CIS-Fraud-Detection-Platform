"""Hyperparameter tuning via Optuna — supports XGBoost, MLP, GNN, and Transformer+GBDT.

Shared design principles
------------------------
- Pipeline fitted ONCE before trials start. Feature engineering is independent
  of model hyperparameters — refitting it 50+ times would be pure waste.
- Each trial logs params + metrics to a nested MLflow child run.
- On completion, best params are written back to model_config.yaml so
  train.py picks them up with no code change (YAML = single source of truth).

XGBoost — four-step pipeline (_run_xgb_pipeline)
-------------------------------------------------
Step 1 — Initial HPO (all features, recency-weighted CV)
    Bayesian optimisation over expanding-window temporal CV folds.
    Objective: weighted mean FPR, recent folds weighted higher (cv_fold_weights)
    to incentivise capturing current fraud patterns.

Step 2 — RFE (stability-weighted feature selection)
    Iteratively removes weakest features by XGBoost importance.
    Selection criterion: minimise mean_FPR + k*std_FPR (UCB score).
    Equivalent to maximising mean−k*std on a recall metric — rewards
    consistent performance over lucky single-fold gains.
    High operational cost of features (engineering, storage, real-time
    compute) makes stability-weighted selection essential.

Step 3 — Re-tune HPO (selected features, recency-weighted CV)
    Re-runs Bayesian optimisation on the reduced feature set.
    Necessary because optimal regularisation and tree depth shift when
    feature count drops — step-1 params on fewer features risk underfitting.

Step 4 — Stability validation
    var(fold FPRs) < variance_threshold (default 0.03).
    Logs pass/fail to MLflow; emits WARNING if failed without aborting.

Neural hybrid models (MLP, GNN, Transformer) — two-phase pipeline (_run_hybrid_pipeline)
------------------------------------------------------------------------------------------
Phase A — Encoder HPO (OOT evaluation, encoder params only)
    Search space is encoder architecture only. XGBoost stage is fixed to base config params
    so tuning budget focuses on encoder quality. Metric: OOT FPR of the full two-stage model.
    After HPO, the best encoder is retrained on the full dev set (A3). Encoder is then frozen.

Phase B — XGBoost optimisation on frozen embeddings
    B1: Extract [original_features || CLS/GNN_embeddings] combined matrix once.
    B2: Level-2 RFE on combined matrix — expanding CV, UCB criterion (same as XGBoost pipeline).
         Removes original features made redundant by embeddings, and weak embedding dims.
    B3: XGBoost re-HPO on selected combined features (recency-weighted expanding CV).
    B4: Stability gate — var(fold FPRs) < variance_threshold.

Usage
-----
    make tune                                      # XGBoost, 50 trials
    make tune MODEL=mlp_xgboost TRIALS=30          # MLP→XGBoost, 30 trials
    make tune MODEL=gnn TRIALS=20                  # GNN, 20 trials
    make tune MODEL=transformer_xgboost            # Transformer+XGBoost
    make tune-then-train                      # tune → update YAML → retrain
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import mlflow
import numpy as np
import optuna
import torch
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config import load_config
from src.data_prep.data_loader import prepare_data
from src.features.build_features import build_features, get_full_pipeline
from src.models.gnn_tree import extract_gnn_embeddings, train_gnn_xgboost
from src.models.mlp_tree import extract_mlp_embeddings, train_mlp_xgboost
from src.models.transformer_tree import (
    extract_embeddings as extract_transformer_embeddings,
    train_transformer_xgboost,
)
from src.models.tree_models import make_fpr_eval_metric
from src.train import time_consistency_split

SECONDS_IN_MONTH = 2_592_000

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.yaml"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def _sample_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """XGBoost search space. learning_rate and regularisation use log-scale
    because their effect spans orders of magnitude."""
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "tree_method":      "hist",
    }



def _sample_gnn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """GNN→XGBoost search space.

    GNN params control the graph encoder (Stage 1).
    XGBoost params control the second-stage classifier (Stage 2).
    """
    return {
        # ---- GNN encoder ----
        "hidden_dim":         trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers":         trial.suggest_int("num_layers", 2, 3),
        "dropout_rate":       trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size":         trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "num_neighbors":      trial.suggest_categorical("num_neighbors", [5, 10, 20]),
        "time_window_days":   trial.suggest_categorical("time_window_days", [3.0, 7.0, 14.0]),
        "max_edges_per_node": trial.suggest_categorical("max_edges_per_node", [5, 10, 15]),
        "epochs":             50,
        "patience":           7,
        # ---- XGBoost (second stage) ----
        "n_estimators":   trial.suggest_int("xgb_n_estimators", 100, 600),
        "max_depth":      trial.suggest_int("xgb_max_depth", 3, 9),
        "subsample":      trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "xgb_early_stopping_rounds": 30,
        "tree_method":    "hist",
    }


def _sample_mlp_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """MLP→XGBoost search space.
    Encoder params: hidden_dims architecture, dropout_rate, learning_rate.
    XGBoost params: same xgb_* space as standalone XGBoost tuning.
    """
    hidden_size = trial.suggest_categorical("hidden_size", ["small", "medium", "large"])
    dims_map = {
        "small":  [256, 128, 64],
        "medium": [512, 256, 128],
        "large":  [512, 256, 128, 64],
    }
    return {
        # Encoder
        "hidden_dims":    dims_map[hidden_size],
        "dropout_rate":   trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "encoder_epochs": trial.suggest_int("encoder_epochs", 10, 30),
        "batch_size":     trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "patience":       5,
        # XGBoost (second stage)
        "n_estimators":   trial.suggest_int("xgb_n_estimators", 100, 600),
        "max_depth":      trial.suggest_int("xgb_max_depth", 3, 9),
        "subsample":      trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "xgb_early_stopping_rounds": 30,
        "tree_method":    "hist",
    }


def _sample_transformer_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Transformer+GBDT search space.
    Encoder params: d_token, nhead, num_layers, dropout_rate, learning_rate.
    XGBoost params: the same xgb_* space as standalone XGBoost tuning.
    nhead must divide d_token evenly — choices are constrained accordingly.
    """
    d_token = trial.suggest_categorical("d_token", [32, 64, 128])
    # nhead must be a divisor of d_token
    valid_nheads = [h for h in [2, 4, 8] if d_token % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_nheads)
    return {
        # Encoder
        "d_token":          d_token,
        "nhead":            nhead,
        "num_layers":       trial.suggest_int("num_layers", 1, 3),
        "dropout_rate":     trial.suggest_float("dropout_rate", 0.05, 0.3),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "encoder_epochs":   30,
        "batch_size":       trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "patience":         5,
        # XGBoost (second stage)
        "n_estimators":     trial.suggest_int("xgb_n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("xgb_max_depth", 3, 9),
        "subsample":        trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "xgb_early_stopping_rounds": 30,
        "tree_method":      "hist",
    }


# ---------------------------------------------------------------------------
# Encoder-only search spaces (Phase A — hybrid pipeline)
# XGBoost stage is fixed to base config params; only encoder architecture is tuned.
# ---------------------------------------------------------------------------

def _sample_mlp_encoder_params(trial: optuna.Trial) -> Dict[str, Any]:
    """MLP encoder-only search space for Phase A HPO.

    hidden_size is a categorical proxy for the hidden_dims list.
    _resolve_encoder_params() converts it to the actual list before training.
    """
    return {
        "hidden_size":    trial.suggest_categorical("hidden_size",
                                                    ["small", "medium", "large"]),
        "dropout_rate":   trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "encoder_epochs": trial.suggest_int("encoder_epochs", 10, 30),
        "batch_size":     trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "patience":       5,
    }


def _sample_transformer_encoder_params(trial: optuna.Trial) -> Dict[str, Any]:
    """TabTransformer encoder-only search space for Phase A HPO."""
    d_token = trial.suggest_categorical("d_token", [32, 64, 128])
    valid_nheads = [h for h in [2, 4, 8] if d_token % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_nheads)
    return {
        "d_token":          d_token,
        "nhead":            nhead,
        "num_layers":       trial.suggest_int("num_layers", 1, 3),
        "dropout_rate":     trial.suggest_float("dropout_rate", 0.05, 0.3),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "encoder_epochs":   30,
        "batch_size":       trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "patience":         5,
    }


def _sample_gnn_encoder_params(trial: optuna.Trial) -> Dict[str, Any]:
    """GNN encoder-only search space for Phase A HPO."""
    return {
        "hidden_dim":         trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers":         trial.suggest_int("num_layers", 2, 3),
        "dropout_rate":       trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size":         trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "num_neighbors":      trial.suggest_categorical("num_neighbors", [5, 10, 20]),
        "time_window_days":   trial.suggest_categorical("time_window_days",
                                                        [3.0, 7.0, 14.0]),
        "max_edges_per_node": trial.suggest_categorical("max_edges_per_node", [5, 10, 15]),
        "epochs":             50,
        "patience":           7,
    }


def _resolve_encoder_params(model_type: str, best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Optuna best_params to actual model params.

    MLP: hidden_size categorical string → hidden_dims list.
    Other models: returned as-is.
    """
    params = dict(best_params)
    if model_type == "mlp_xgboost" and "hidden_size" in params:
        dims_map = {
            "small":  [256, 128, 64],
            "medium": [512, 256, 128],
            "large":  [512, 256, 128, 64],
        }
        params["hidden_dims"] = dims_map[params.pop("hidden_size")]
    return params


# ---------------------------------------------------------------------------
# Temporal CV helpers (XGBoost only)
# ---------------------------------------------------------------------------

def _make_temporal_cv_folds(
    month_arr: np.ndarray, n_folds: int = 3
) -> List[tuple]:
    """Expanding-window temporal CV splits over a month index array.

    With n_folds=3 and months [0..5]:
      Fold 1: train months [0-2], val month 3
      Fold 2: train months [0-3], val month 4
      Fold 3: train months [0-4], val month 5

    Returns list of (train_indices, val_indices) — integer positions into
    month_arr (i.e. into the already-processed feature matrix).
    """
    unique_months = np.sort(np.unique(month_arr))
    n = len(unique_months)
    folds = []
    for i in range(n_folds):
        val_pos = n - n_folds + i
        if val_pos <= 0:
            continue
        val_month = unique_months[val_pos]
        train_months = unique_months[:val_pos]
        train_idx = np.where(np.isin(month_arr, train_months))[0]
        val_idx = np.where(month_arr == val_month)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds


def _cv_xgb_fold_fprs(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: List[tuple],
    params: Dict[str, Any],
    fpr_threshold: float,
    early_stopping_rounds: int = 50,
    feature_idx: Optional[np.ndarray] = None,
) -> List[float]:
    """Run temporal CV and return per-fold FPR list (same order as cv_folds).

    Returning per-fold values lets callers compute any aggregation:
    weighted mean (HPO), mean+k*std (RFE), variance (stability gate).

    eval_metric and early_stopping_rounds are stripped from params and
    replaced with the FPR metric so callers do not need to sanitise first.
    """
    clean_params = {k: v for k, v in params.items()
                    if k not in ("eval_metric", "early_stopping_rounds")}
    X_sel = X[:, feature_idx] if feature_idx is not None else X
    fold_fprs = []
    for train_idx, val_idx in cv_folds:
        model = XGBClassifier(
            **clean_params,
            eval_metric=make_fpr_eval_metric(fpr_threshold),
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
        )
        model.fit(
            X_sel[train_idx], y[train_idx],
            eval_set=[(X_sel[val_idx], y[val_idx])],
            verbose=False,
        )
        probs = model.predict_proba(X_sel[val_idx])[:, 1]
        preds_binary = (probs >= fpr_threshold).astype(int)
        negatives = y[val_idx] == 0
        fp = int(((preds_binary == 1) & negatives).sum())
        tn = int(((preds_binary == 0) & negatives).sum())
        fold_fprs.append(fp / (fp + tn + 1e-8))
    return fold_fprs


def _weighted_mean_fpr(fold_fprs: List[float], weights: List[float]) -> float:
    """Weighted mean FPR — recent folds carry higher weight.

    weights need not be normalised; any positive values work.
    E.g. weights=[1,2,3] for 3 folds gives fold-3 (most recent) 3x the weight.
    """
    w = np.array(weights, dtype=float)
    return float(np.dot(w / w.sum(), fold_fprs))


def _rfe_score(fold_fprs: List[float], k: float = 2.0) -> float:
    """Upper-confidence-bound FPR = mean + k*std.

    Minimising this selects the feature set that is both low-FPR on average
    AND stable across folds. This is the FPR-direction equivalent of the
    'highest mean - k*std' criterion on a recall/AUC metric: penalising high
    variance prevents selecting a feature set that lucked out on a single fold.
    """
    arr = np.array(fold_fprs)
    return float(arr.mean() + k * arr.std())


def _run_xgb_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: List[tuple],
    cv_fold_weights: List[float],
    early_stopping_rounds: int,
    fpr_threshold: float,
    n_trials: int,
    study_name: str,
    parent_run_id: str,
    feature_idx: Optional[np.ndarray] = None,
) -> optuna.Study:
    """Create and optimise one Optuna study. Shared by Step 1 and Step 3."""
    objective = _build_xgb_objective(
        X_train, y_train, cv_folds, cv_fold_weights,
        early_stopping_rounds, fpr_threshold, parent_run_id,
        feature_idx=feature_idx,
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    logger.info("Optuna study '%s': %d trials on %s features",
                study_name, n_trials,
                X_train.shape[1] if feature_idx is None else len(feature_idx))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _rfe_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: List[tuple],
    best_params: Dict[str, Any],
    fpr_threshold: float,
    n_features_min: int = 20,
    step: int = 10,
    early_stopping_rounds: int = 50,
    stability_k: float = 2.0,
) -> List[int]:
    """Recursive Feature Elimination for XGBoost.

    Selection criterion: minimise mean_FPR + stability_k * std_FPR across folds
    (upper-confidence-bound on FPR). This jointly prefers low average FPR and
    low variance — a feature set that barely beats the threshold on one fold
    but fails on others is penalised more than a consistently good set.

    Each iteration:
      1. Score current feature set via CV using the UCB criterion.
      2. Fit on full training data to rank features by importance.
      3. Drop the `step` weakest features.
    Stops when dropping another batch would go below n_features_min.
    Returns the feature-index list with the lowest UCB score seen.
    """
    clean_params = {k: v for k, v in best_params.items()
                    if k not in ("eval_metric", "early_stopping_rounds")}
    feature_idx = np.arange(X_train.shape[1])
    best_score = np.inf
    best_feature_idx = feature_idx.copy()

    logger.info("RFE: %d → %d features, step=%d, stability_k=%.1f",
                X_train.shape[1], n_features_min, step, stability_k)

    while len(feature_idx) > n_features_min:
        fold_fprs = _cv_xgb_fold_fprs(
            X_train, y_train, cv_folds, clean_params, fpr_threshold,
            early_stopping_rounds, feature_idx=feature_idx,
        )
        mean_fpr = float(np.mean(fold_fprs))
        std_fpr  = float(np.std(fold_fprs))
        score    = _rfe_score(fold_fprs, k=stability_k)

        logger.info("RFE: %d features — mean_FPR=%.4f  std=%.4f  ucb_score=%.4f",
                    len(feature_idx), mean_fpr, std_fpr, score)

        if score <= best_score:
            best_score = score
            best_feature_idx = feature_idx.copy()

        # Fit on full training data (no early stopping) to rank importances
        importance_model = XGBClassifier(**clean_params, verbosity=0)
        importance_model.fit(X_train[:, feature_idx], y_train, verbose=False)
        importances = importance_model.feature_importances_

        remove_n = min(step, len(feature_idx) - n_features_min)
        weakest_local = np.argsort(importances)[:remove_n]
        feature_idx = np.delete(feature_idx, weakest_local)

    logger.info("RFE complete: selected %d features, best UCB score=%.4f",
                len(best_feature_idx), best_score)
    return best_feature_idx.tolist()


def _run_xgb_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    month_arr: np.ndarray,
    fpr_threshold: float,
    model_tuning: Dict[str, Any],
    n_trials: int,
    parent_run_id: str,
) -> tuple:
    """Four-step XGBoost training pipeline.

    Step 1 — Initial HPO (all features, weighted CV)
        Bayesian optimisation minimising recency-weighted mean FPR.
        Recent folds are weighted higher to incentivise capturing current
        fraud patterns over historical ones.

    Step 2 — RFE (feature selection, stability criterion)
        Iteratively drops weakest features. Selects the subset that minimises
        mean_FPR + k*std_FPR (UCB), rewarding consistent performance across
        folds. High operational cost of features (engineering, storage,
        real-time compute) makes stability-weighted selection essential.

    Step 3 — Re-tune HPO (selected features, weighted CV)
        Re-runs Bayesian optimisation on the reduced feature set. Necessary
        because optimal regularisation and tree complexity shift when the
        feature count drops — using step-1 params on fewer features risks
        underfitting.

    Step 4 — Stability validation
        Checks that var(fold FPRs) < variance_threshold. A model that passes
        steps 1-3 but shows high fold-to-fold variance is not safe to promote:
        it will produce inconsistent live scores. Logs pass/fail to MLflow and
        emits a warning without aborting (caller decides on promotion).

    Returns (best_params, selected_feature_indices, final_study).
    """
    early_stopping_rounds = model_tuning.get("early_stopping_rounds", 50)
    n_cv_folds            = model_tuning.get("cv_folds", 3)
    cv_fold_weights       = model_tuning.get("cv_fold_weights",
                                             list(range(1, n_cv_folds + 1)))
    rfe_stability_k       = model_tuning.get("rfe_stability_k", 2.0)
    variance_threshold    = model_tuning.get("variance_threshold", 0.03)
    n_features_min        = model_tuning.get("rfe_n_features_min", 20)
    rfe_step              = model_tuning.get("rfe_step", 10)
    retune_n_trials       = model_tuning.get("retune_n_trials",
                                             max(n_trials // 2, 20))
    base_study_name       = model_tuning.get("study_name", "xgboost_fraud_tuning")

    cv_folds = _make_temporal_cv_folds(month_arr, n_folds=n_cv_folds)
    logger.info("Temporal CV: %d folds, weights=%s", len(cv_folds), cv_fold_weights)
    mlflow.log_param("cv_folds", len(cv_folds))
    mlflow.log_param("cv_fold_weights", str(cv_fold_weights))

    # ------------------------------------------------------------------
    # Step 1: Initial HPO — all features, weighted CV
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Initial HPO on all %d features ===", X_train.shape[1])
    study_1 = _run_xgb_study(
        X_train, y_train, cv_folds, cv_fold_weights,
        early_stopping_rounds, fpr_threshold,
        n_trials=n_trials,
        study_name=base_study_name,
        parent_run_id=parent_run_id,
        feature_idx=None,
    )
    best_params_1 = dict(study_1.best_params)
    best_params_1["tree_method"] = "hist"
    best_params_1["eval_metric"] = "auc"
    mlflow.log_metric("step1_best_weighted_FPR", study_1.best_value)
    mlflow.set_tag("step1_best_trial", study_1.best_trial.number)
    logger.info("Step 1 complete. Best weighted FPR=%.4f (trial %d)",
                study_1.best_value, study_1.best_trial.number)

    # ------------------------------------------------------------------
    # Step 2: RFE — stability-weighted feature selection
    # ------------------------------------------------------------------
    logger.info("=== Step 2: RFE (min_features=%d, step=%d, k=%.1f) ===",
                n_features_min, rfe_step, rfe_stability_k)
    selected_features = _rfe_xgboost(
        X_train, y_train, cv_folds, best_params_1,
        fpr_threshold, n_features_min, rfe_step,
        early_stopping_rounds, stability_k=rfe_stability_k,
    )
    feat_idx = np.array(selected_features)
    mlflow.log_param("n_selected_features", len(selected_features))
    logger.info("Step 2 complete. Selected %d features.", len(selected_features))

    # ------------------------------------------------------------------
    # Step 3: Re-tune HPO — selected features, weighted CV
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Re-tune HPO on %d features (%d trials) ===",
                len(selected_features), retune_n_trials)
    study_3 = _run_xgb_study(
        X_train, y_train, cv_folds, cv_fold_weights,
        early_stopping_rounds, fpr_threshold,
        n_trials=retune_n_trials,
        study_name=base_study_name + "_retune",
        parent_run_id=parent_run_id,
        feature_idx=feat_idx,
    )
    best_params_final = dict(study_3.best_params)
    best_params_final["tree_method"] = "hist"
    best_params_final["eval_metric"] = "auc"
    mlflow.log_metric("step3_best_weighted_FPR", study_3.best_value)
    mlflow.set_tag("step3_best_trial", study_3.best_trial.number)
    logger.info("Step 3 complete. Best weighted FPR=%.4f (trial %d)",
                study_3.best_value, study_3.best_trial.number)

    # ------------------------------------------------------------------
    # Step 4: Stability validation — var(fold FPRs) < threshold
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Stability validation (threshold=%.3f) ===",
                variance_threshold)
    final_fold_fprs = _cv_xgb_fold_fprs(
        X_train, y_train, cv_folds, best_params_final,
        fpr_threshold, early_stopping_rounds, feature_idx=feat_idx,
    )
    cv_mean     = float(np.mean(final_fold_fprs))
    cv_std      = float(np.std(final_fold_fprs))
    cv_variance = float(np.var(final_fold_fprs))

    mlflow.log_metric("final_CV_FPR_mean",     cv_mean)
    mlflow.log_metric("final_CV_FPR_std",      cv_std)
    mlflow.log_metric("final_CV_FPR_variance", cv_variance)
    for i, fpr in enumerate(final_fold_fprs):
        mlflow.log_metric(f"final_fold_{i}_FPR", fpr)

    stability_ok = cv_variance < variance_threshold
    mlflow.set_tag("stability_gate_passed", str(stability_ok))
    if stability_ok:
        logger.info("Step 4 PASSED: variance=%.4f < %.3f  (mean=%.4f  std=%.4f)",
                    cv_variance, variance_threshold, cv_mean, cv_std)
    else:
        logger.warning(
            "Step 4 FAILED: variance=%.4f >= %.3f  (mean=%.4f  std=%.4f). "
            "Model output is unstable across CV folds — review feature set "
            "or lower rfe_stability_k before promoting to production.",
            cv_variance, variance_threshold, cv_mean, cv_std,
        )

    return best_params_final, selected_features, study_3


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _build_xgb_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: List[tuple],
    cv_fold_weights: List[float],
    early_stopping_rounds: int,
    fpr_threshold: float,
    parent_run_id: str,
    feature_idx: Optional[np.ndarray] = None,
):
    """Optuna objective for XGBoost.

    Minimises recency-weighted mean FPR across temporal CV folds.
    cv_fold_weights[i] is the weight for fold i (fold 0 = oldest).
    Higher weights on recent folds incentivise capturing current fraud patterns.
    feature_idx restricts evaluation to a feature subset (used in Step 3 re-tune).
    """
    def objective(trial: optuna.Trial) -> float:
        params = _sample_xgb_params(trial)
        with mlflow.start_run(run_id=parent_run_id, nested=False):
            with mlflow.start_run(run_name=f"xgb_trial_{trial.number:04d}", nested=True):
                mlflow.log_params({k: v for k, v in params.items() if k != "tree_method"})
                mlflow.set_tag("trial_number", trial.number)

                fold_fprs = _cv_xgb_fold_fprs(
                    X_train, y_train, cv_folds, params,
                    fpr_threshold, early_stopping_rounds,
                    feature_idx=feature_idx,
                )
                weighted_fpr = _weighted_mean_fpr(fold_fprs, cv_fold_weights)

                mlflow.log_metric("CV_weighted_FPR", weighted_fpr)
                for i, f in enumerate(fold_fprs):
                    mlflow.log_metric(f"fold_{i}_FPR", f)
                logger.info("XGB trial %d — weighted_FPR=%.4f  folds=%s",
                            trial.number, weighted_fpr,
                            [f"{v:.4f}" for v in fold_fprs])
        return weighted_fpr
    return objective



def _build_hybrid_encoder_objective(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fixed_xgb_params: Dict[str, Any],
    fpr_threshold: float,
    parent_run_id: str,
    device: str,
    X_train_raw=None,
    X_test_raw=None,
):
    """Optuna objective for encoder-only HPO (Phase A).

    Search space: encoder architecture params only (hidden_dims/d_token/GNN graph params).
    XGBoost stage uses fixed base config params — its LR is intentionally overridden
    by the encoder's trial LR in the combined dict, which is acceptable here because
    this XGBoost is a scoring proxy only (discarded after Phase A).

    Metric: OOT FPR of the full two-stage model at fpr_threshold.
    """
    # Pre-compute feature-engineered DataFrames for GNN once (graph edges require uid
    # column that only exists after build_features — expensive to recompute per trial).
    if model_type == "gnn":
        X_train_eng = build_features(X_train_raw.copy())
        X_test_eng  = build_features(X_test_raw.copy())

    def objective(trial: optuna.Trial) -> float:
        if model_type == "mlp_xgboost":
            raw_enc_params = _sample_mlp_encoder_params(trial)
        elif model_type == "transformer_xgboost":
            raw_enc_params = _sample_transformer_encoder_params(trial)
        else:  # gnn
            raw_enc_params = _sample_gnn_encoder_params(trial)

        enc_params = _resolve_encoder_params(model_type, raw_enc_params)
        # Encoder params override fixed_xgb_params so the encoder's learning_rate
        # is used by the encoder; the XGBoost proxy picks up enc LR as well, which
        # is acceptable since this XGBoost result is discarded after Phase A.
        combined = {**fixed_xgb_params, **enc_params}

        with mlflow.start_run(run_id=parent_run_id, nested=False):
            with mlflow.start_run(
                run_name=f"{model_type}_enc_trial_{trial.number:04d}", nested=True
            ):
                mlflow.log_params({k: v for k, v in enc_params.items()
                                   if k not in ("epochs", "patience",
                                                "encoder_epochs", "tree_method")})
                mlflow.set_tag("phase", "A_encoder_hpo")
                mlflow.set_tag("trial_number", trial.number)

                if model_type == "mlp_xgboost":
                    encoder, xgb_model = train_mlp_xgboost(
                        X_train, y_train, X_test, y_test,
                        params=combined, fpr_threshold=fpr_threshold,
                        device=device, save_path=None,
                    )
                    embed_test      = extract_mlp_embeddings(encoder, X_test, device=device)
                    X_test_enriched = np.concatenate([X_test, embed_test], axis=1)

                elif model_type == "transformer_xgboost":
                    encoder, xgb_model = train_transformer_xgboost(
                        X_train, y_train, X_test, y_test,
                        params=combined, fpr_threshold=fpr_threshold,
                        device=device, save_path=None,
                    )
                    embed_test      = extract_transformer_embeddings(
                        encoder, X_test, device=device)
                    X_test_enriched = np.concatenate([X_test, embed_test], axis=1)

                else:  # gnn
                    from torch_geometric.data import Data  # noqa: PLC0415
                    from src.models.gnn_tree import build_transaction_graph  # noqa: PLC0415

                    gnn_model, xgb_model = train_gnn_xgboost(
                        X_train, y_train, X_test, y_test,
                        X_train_eng, X_test_eng,
                        params=combined, fpr_threshold=fpr_threshold,
                        device=device, save_path=None,
                    )
                    X_eng_combined = pd.concat(
                        [X_train_eng.reset_index(drop=True),
                         X_test_eng.reset_index(drop=True)], ignore_index=True)
                    X_proc_combined = np.vstack([X_train, X_test])
                    edge_index, edge_attr, x = build_transaction_graph(
                        X_eng_combined, X_proc_combined,
                        max_edges_per_node=combined.get("max_edges_per_node", 10))
                    from torch_geometric.data import Data as TGData  # noqa: PLC0415
                    data = TGData(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    _, test_emb = extract_gnn_embeddings(
                        gnn_model, data, len(X_train), device=device)
                    X_test_enriched = np.concatenate([X_test, test_emb], axis=1)
                    encoder = gnn_model  # uniform name for logging

                probs        = xgb_model.predict_proba(X_test_enriched)[:, 1]
                y_v          = y_test.values if hasattr(y_test, "values") else y_test
                preds_binary = (probs >= fpr_threshold).astype(int)
                negatives    = y_v == 0
                fp           = int(((preds_binary == 1) & negatives).sum())
                tn           = int(((preds_binary == 0) & negatives).sum())
                trial_fpr    = fp / (fp + tn + 1e-8)
                auc          = float(roc_auc_score(y_v, probs))

                mlflow.log_metric("OOT_FPR", trial_fpr)
                mlflow.log_metric("OOT_AUC", auc)
                logger.info("%s encoder trial %d — FPR: %.4f  AUC: %.4f",
                            model_type, trial.number, trial_fpr, auc)
        return trial_fpr

    return objective


def _train_final_encoder(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    combined_params: Dict[str, Any],
    fpr_threshold: float,
    device: str,
    X_train_raw=None,
    X_test_raw=None,
):
    """Train final encoder on the full dev set (Phase A3).

    Calls the integrated two-stage train function but returns only the encoder
    (XGBoost trained here is discarded — Phase B will train its own XGBoost on
    the frozen encoder's combined-feature matrix with tuned Stage-2 params).

    Returns encoder_model (MLPEncoder / TabTransformer / FraudGNN).
    """
    if model_type == "mlp_xgboost":
        encoder, _ = train_mlp_xgboost(
            X_train, y_train, X_test, y_test,
            params=combined_params, fpr_threshold=fpr_threshold,
            device=device, save_path=None,
        )
        return encoder

    elif model_type == "transformer_xgboost":
        encoder, _ = train_transformer_xgboost(
            X_train, y_train, X_test, y_test,
            params=combined_params, fpr_threshold=fpr_threshold,
            device=device, save_path=None,
        )
        return encoder

    else:  # gnn
        X_train_eng = build_features(X_train_raw.copy())
        X_test_eng  = build_features(X_test_raw.copy())
        gnn_model, _ = train_gnn_xgboost(
            X_train, y_train, X_test, y_test,
            X_train_eng, X_test_eng,
            params=combined_params, fpr_threshold=fpr_threshold,
            device=device, save_path=None,
        )
        return gnn_model


def _extract_combined_matrix(
    model_type: str,
    encoder_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_train_raw=None,
    X_test_raw=None,
    device: str = "cpu",
    max_edges_per_node: int = 10,
):
    """Extract [original_features || embeddings] from a frozen encoder.

    GNN requires building the full transaction graph (train+test combined) and
    slicing at split_idx — test nodes see train edges during inference, matching
    how GNN embeddings are computed at serving time.

    Returns (X_combined_train, X_combined_test, n_original_features).
    n_original_features = X_train.shape[1], stored so train.py can reconstruct
    the column split without re-running the encoder.
    """
    n_orig = X_train.shape[1]

    if model_type == "mlp_xgboost":
        embed_train = extract_mlp_embeddings(encoder_model, X_train, device=device)
        embed_test  = extract_mlp_embeddings(encoder_model, X_test,  device=device)

    elif model_type == "transformer_xgboost":
        embed_train = extract_transformer_embeddings(encoder_model, X_train, device=device)
        embed_test  = extract_transformer_embeddings(encoder_model, X_test,  device=device)

    else:  # gnn
        from torch_geometric.data import Data  # noqa: PLC0415
        from src.models.gnn_tree import build_transaction_graph  # noqa: PLC0415

        X_train_eng = build_features(X_train_raw.copy())
        X_test_eng  = build_features(X_test_raw.copy())
        X_eng_comb  = pd.concat(
            [X_train_eng.reset_index(drop=True), X_test_eng.reset_index(drop=True)],
            ignore_index=True,
        )
        X_proc_comb = np.vstack([X_train, X_test])
        edge_index, edge_attr, x = build_transaction_graph(
            X_eng_comb, X_proc_comb, max_edges_per_node=max_edges_per_node,
        )
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        embed_train, embed_test = extract_gnn_embeddings(
            encoder_model, data, len(X_train), device=device,
        )

    X_combined_train = np.concatenate([X_train, embed_train], axis=1)
    X_combined_test  = np.concatenate([X_test,  embed_test],  axis=1)
    return X_combined_train, X_combined_test, n_orig


# ---------------------------------------------------------------------------
# Hybrid pipeline (MLP / GNN / Transformer)
# ---------------------------------------------------------------------------

def _run_hybrid_pipeline(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_raw,
    X_test_raw,
    month_arr: np.ndarray,
    fpr_threshold: float,
    model_tuning: Dict[str, Any],
    n_trials: int,
    parent_run_id: str,
    device: str,
    cfg: Dict[str, Any],
) -> tuple:
    """Two-phase hybrid tuning pipeline for MLP/GNN/Transformer models.

    Phase A — Encoder HPO + final training
        A1: Bayesian HPO over encoder params only (OOT FPR objective).
            XGBoost stage uses fixed base config params as a scoring proxy.
        A2: Resolve best encoder params (e.g. hidden_size → hidden_dims for MLP).
        A3: Retrain encoder on full dev set with best params. Encoder is then frozen.

    Phase B — XGBoost optimisation on frozen embeddings
        B0: Extract [original || embeddings] combined matrix once.
        B1: Level-2 RFE on combined matrix (expanding CV, UCB stability criterion).
        B2: XGBoost re-HPO on selected combined features (recency-weighted CV).
        B3: Stability gate — var(fold FPRs) < variance_threshold.

    Returns:
        (best_encoder_params, best_xgb_stage2_params, l2_combined_indices,
         n_original_features, final_study)
    """
    # ---- Config ----
    encoder_n_trials   = model_tuning.get("encoder_n_trials", n_trials)
    n_cv_folds         = model_tuning.get("cv_folds", 3)
    cv_fold_weights    = model_tuning.get("cv_fold_weights",
                                          list(range(1, n_cv_folds + 1)))
    early_stopping_rds = model_tuning.get("early_stopping_rounds", 50)
    rfe_stability_k    = model_tuning.get("rfe_stability_k", 2.0)
    variance_threshold = model_tuning.get("variance_threshold", 0.03)
    n_features_min     = model_tuning.get("rfe_n_features_min", 20)
    rfe_step           = model_tuning.get("rfe_step", 10)
    retune_n_trials    = model_tuning.get("retune_n_trials", max(n_trials // 2, 20))
    base_study_name    = model_tuning.get("study_name", f"{model_type}_fraud_tuning")

    # Fixed XGBoost params used as Stage-2 proxy during Phase A.
    # These are base config params (n_estimators=500, etc.) — sufficient to score
    # encoder quality without spending budget on XGBoost optimisation here.
    fixed_xgb_params = cfg.get("xgboost_params", {})

    # ==========================================================================
    # Phase A — Encoder HPO
    # ==========================================================================
    logger.info("=== Phase A: Encoder HPO — %d trials, OOT FPR objective ===",
                encoder_n_trials)
    enc_objective = _build_hybrid_encoder_objective(
        model_type, X_train, y_train, X_test, y_test,
        fixed_xgb_params=fixed_xgb_params,
        fpr_threshold=fpr_threshold,
        parent_run_id=parent_run_id,
        device=device,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
    )
    enc_study = optuna.create_study(
        direction="minimize",
        study_name=base_study_name + "_encoder",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    enc_study.optimize(enc_objective, n_trials=encoder_n_trials, show_progress_bar=True)

    best_encoder_params = _resolve_encoder_params(model_type,
                                                   enc_study.best_params)
    mlflow.log_metric("phaseA_best_OOT_FPR", enc_study.best_value)
    mlflow.set_tag("phaseA_best_trial", enc_study.best_trial.number)
    logger.info("Phase A complete. Best OOT FPR=%.4f (trial %d)",
                enc_study.best_value, enc_study.best_trial.number)

    # ==========================================================================
    # Phase A3 — Final encoder training on full dev set
    # ==========================================================================
    logger.info("=== Phase A3: Final encoder training (full dev set) ===")
    combined_for_a3 = {**fixed_xgb_params, **best_encoder_params}
    encoder_model = _train_final_encoder(
        model_type, X_train, y_train, X_test, y_test,
        combined_params=combined_for_a3,
        fpr_threshold=fpr_threshold,
        device=device,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
    )
    logger.info("Phase A3 complete. Encoder frozen.")

    # ==========================================================================
    # Phase B — XGBoost on frozen combined-feature matrix
    # ==========================================================================
    logger.info("=== Phase B: Extract embeddings → L2 RFE → XGBoost re-HPO ===")

    # B0: Extract [original || embeddings] once
    X_comb_train, X_comb_test, n_original_features = _extract_combined_matrix(
        model_type, encoder_model, X_train, X_test,
        X_train_raw, X_test_raw, device,
        max_edges_per_node=best_encoder_params.get("max_edges_per_node", 10),
    )
    logger.info(
        "Combined matrix: train=%s  test=%s  (orig=%d  embed=%d)",
        X_comb_train.shape, X_comb_test.shape,
        n_original_features, X_comb_train.shape[1] - n_original_features,
    )
    mlflow.log_param("phaseB_combined_features",    X_comb_train.shape[1])
    mlflow.log_param("phaseB_n_original_features",  n_original_features)

    cv_folds = _make_temporal_cv_folds(month_arr, n_folds=n_cv_folds)
    logger.info("Phase B CV: %d folds, weights=%s", len(cv_folds), cv_fold_weights)

    # Initial XGBoost params for RFE (fixed base params — we just need importances)
    rfe_xgb_params = {
        "n_estimators":   fixed_xgb_params.get("n_estimators", 300),
        "learning_rate":  fixed_xgb_params.get("learning_rate", 0.05),
        "max_depth":      fixed_xgb_params.get("max_depth", 6),
        "subsample":      fixed_xgb_params.get("subsample", 0.8),
        "colsample_bytree": fixed_xgb_params.get("colsample_bytree", 0.8),
        "tree_method": "hist",
    }

    # B1: Level-2 RFE on combined matrix
    logger.info("=== Phase B1: L2 RFE (min=%d, step=%d, k=%.1f) ===",
                n_features_min, rfe_step, rfe_stability_k)
    l2_combined_indices = _rfe_xgboost(
        X_comb_train, y_train, cv_folds, rfe_xgb_params,
        fpr_threshold, n_features_min, rfe_step,
        early_stopping_rds, stability_k=rfe_stability_k,
    )
    mlflow.log_param("phaseB_n_selected", len(l2_combined_indices))
    logger.info("Phase B1 complete. Selected %d combined features.",
                len(l2_combined_indices))

    # B2: XGBoost re-HPO on selected combined features
    l2_feat_idx = np.array(l2_combined_indices)
    logger.info("=== Phase B2: XGBoost re-HPO — %d features, %d trials ===",
                len(l2_combined_indices), retune_n_trials)
    stage2_study = _run_xgb_study(
        X_comb_train, y_train, cv_folds, cv_fold_weights,
        early_stopping_rds, fpr_threshold,
        n_trials=retune_n_trials,
        study_name=base_study_name + "_stage2",
        parent_run_id=parent_run_id,
        feature_idx=l2_feat_idx,
    )
    best_xgb_stage2_params = dict(stage2_study.best_params)
    best_xgb_stage2_params["tree_method"] = "hist"
    best_xgb_stage2_params["eval_metric"] = "auc"
    mlflow.log_metric("phaseB2_best_weighted_FPR", stage2_study.best_value)
    mlflow.set_tag("phaseB2_best_trial", stage2_study.best_trial.number)
    logger.info("Phase B2 complete. Best weighted FPR=%.4f (trial %d)",
                stage2_study.best_value, stage2_study.best_trial.number)

    # B3: Stability gate
    logger.info("=== Phase B3: Stability gate (threshold=%.3f) ===", variance_threshold)
    final_fold_fprs = _cv_xgb_fold_fprs(
        X_comb_train, y_train, cv_folds, best_xgb_stage2_params,
        fpr_threshold, early_stopping_rds, feature_idx=l2_feat_idx,
    )
    cv_mean     = float(np.mean(final_fold_fprs))
    cv_std      = float(np.std(final_fold_fprs))
    cv_variance = float(np.var(final_fold_fprs))

    mlflow.log_metric("phaseB_final_CV_FPR_mean",     cv_mean)
    mlflow.log_metric("phaseB_final_CV_FPR_std",      cv_std)
    mlflow.log_metric("phaseB_final_CV_FPR_variance", cv_variance)
    for i, fpr_val in enumerate(final_fold_fprs):
        mlflow.log_metric(f"phaseB_final_fold_{i}_FPR", fpr_val)

    stability_ok = cv_variance < variance_threshold
    mlflow.set_tag("phaseB_stability_gate_passed", str(stability_ok))
    if stability_ok:
        logger.info("Phase B3 PASSED: var=%.4f < %.3f  (mean=%.4f  std=%.4f)",
                    cv_variance, variance_threshold, cv_mean, cv_std)
    else:
        logger.warning(
            "Phase B3 FAILED: var=%.4f >= %.3f  (mean=%.4f  std=%.4f). "
            "Review feature set or lower rfe_stability_k before promoting.",
            cv_variance, variance_threshold, cv_mean, cv_std,
        )

    return (best_encoder_params, best_xgb_stage2_params,
            l2_combined_indices, n_original_features, stage2_study)


# ---------------------------------------------------------------------------
# Config write-back
# ---------------------------------------------------------------------------

def _write_best_params_to_config(
    best_params: Dict[str, Any],
    model_type: str,
    config_path: Optional[str] = None,
    selected_features: Optional[List[int]] = None,
    stage2_params: Optional[Dict[str, Any]] = None,
    l2_indices: Optional[List[int]] = None,
) -> None:
    """Overwrite the model-specific params block in model_config.yaml.

    XGBoost (four-step pipeline):
        xgboost_params            ← best_params (Step 3 re-tuned)
        xgboost_selected_features ← selected_features (Step 2 RFE output)

    Hybrid models (two-phase pipeline):
        {model}_params      ← best_params (Phase A encoder params)
        {model}_stage2_params ← stage2_params (Phase B XGBoost params)
        {model}_selected      ← l2_indices (Phase B L2 RFE combined indices)

    All other YAML sections are preserved. train.py reads these keys to
    reconstruct the full two-stage model with no code change.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    yaml_key_map = {
        "xgboost":             "xgboost_params",
        "mlp_xgboost":         "mlp_xgboost_params",
        "gnn":                 "gnn_params",
        "transformer_xgboost": "transformer_params",
    }
    yaml_key = yaml_key_map[model_type]

    skip_keys = {"use_label_encoder", "epochs", "patience", "encoder_epochs",
                 "xgb_early_stopping_rounds", "tree_method"}
    cfg[yaml_key] = {k: v for k, v in best_params.items() if k not in skip_keys}

    # XGBoost-only: L1 RFE selected features
    if selected_features is not None:
        cfg["xgboost_selected_features"] = selected_features

    # Hybrid models: Phase B XGBoost stage-2 params and L2 combined indices
    stage2_key_map = {
        "mlp_xgboost":         "mlp_xgboost_stage2_params",
        "gnn":                 "gnn_stage2_params",
        "transformer_xgboost": "transformer_stage2_params",
    }
    l2_key_map = {
        "mlp_xgboost":         "mlp_xgboost_selected",
        "gnn":                 "gnn_selected",
        "transformer_xgboost": "transformer_xgboost_selected",
    }
    if stage2_params is not None:
        s2_key = stage2_key_map.get(model_type)
        if s2_key:
            cfg[s2_key] = {k: v for k, v in stage2_params.items()
                           if k not in skip_keys}
    if l2_indices is not None:
        l2_key = l2_key_map.get(model_type)
        if l2_key:
            cfg[l2_key] = l2_indices

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.info("Best %s params written to %s[%s]", model_type, path, yaml_key)
    if selected_features is not None:
        logger.info("XGBoost L1 RFE: %d features → xgboost_selected_features",
                    len(selected_features))
    if l2_indices is not None:
        logger.info("Hybrid L2 RFE: %d combined indices → %s",
                    len(l2_indices), l2_key_map.get(model_type, "?"))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

VALID_MODELS = ("xgboost", "mlp_xgboost", "gnn", "transformer_xgboost")


def run_tuning(
    trans_path: str,
    id_path: str,
    config_path: Optional[str] = None,
    n_trials: int = 50,
    model_type: str = "xgboost",
) -> optuna.Study:
    """Orchestrate the full tuning run for any supported model type.

    XGBoost — four-step pipeline (_run_xgb_pipeline):
        1. Initial HPO — Bayesian opt, recency-weighted expanding CV, all features.
        2. RFE — UCB stability criterion, expanding CV.
        3. Re-tune HPO — selected features, recency-weighted CV.
        4. Stability gate — var(fold FPRs) < variance_threshold.

    MLP / GNN / Transformer — two-phase hybrid pipeline (_run_hybrid_pipeline):
        Phase A: Encoder HPO (OOT FPR, encoder params only) + final encoder training.
        Phase B: Extract [orig || embeddings], L2 RFE, XGBoost re-HPO, stability gate.
    """
    if model_type not in VALID_MODELS:
        raise ValueError(f"model_type must be one of {VALID_MODELS}, got '{model_type}'")

    cfg          = load_config(config_path)
    training_cfg = cfg["training"]
    tuning_cfg   = cfg.get("tuning", {})
    model_tuning = tuning_cfg.get(model_type, {})
    fpr_threshold = cfg["serving"]["fraud_threshold_prob"]

    mlflow.set_tracking_uri(training_cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(training_cfg["mlflow_experiment_name"])

    logger.info("Loading and preparing data...")
    X, y = prepare_data(trans_path, id_path)
    train_idx, test_idx = time_consistency_split(X)

    X_train_raw, y_train = X.loc[train_idx], y.loc[train_idx].values
    X_test_raw,  y_test  = X.loc[test_idx],  y.loc[test_idx].values

    logger.info("Fitting feature pipeline once on %d training samples...", len(X_train_raw))
    pipeline = get_full_pipeline()
    X_train  = pipeline.fit_transform(X_train_raw)
    X_test   = pipeline.transform(X_test_raw)
    logger.info("Pipeline fitted. train=%s  test=%s", X_train.shape, X_test.shape)

    study_name = model_tuning.get("study_name", f"{model_type}_fraud_tuning")
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    with mlflow.start_run(run_name=study_name) as parent_run:
        mlflow.set_tag("model_type",       model_type)
        mlflow.set_tag("tuning_framework", "optuna")
        mlflow.set_tag("sampler",          "TPESampler")
        mlflow.set_tag("n_trials",         n_trials)
        mlflow.log_param("train_samples",  X_train.shape[0])
        mlflow.log_param("test_samples",   X_test.shape[0])
        mlflow.log_param("n_features",     X_train.shape[1])
        mlflow.log_param("fpr_threshold",  fpr_threshold)

        month_arr = np.floor(
            X_train_raw["TransactionDT"].values / SECONDS_IN_MONTH
        )

        if model_type == "xgboost":
            best, selected_features, study = _run_xgb_pipeline(
                X_train, y_train, month_arr,
                fpr_threshold, model_tuning, n_trials,
                parent_run_id=parent_run.info.run_id,
            )
            _write_best_params_to_config(
                best, model_type, config_path,
                selected_features=selected_features,
            )

        else:  # mlp_xgboost, gnn, transformer_xgboost — two-phase hybrid pipeline
            (best_encoder_params, best_xgb_stage2_params,
             l2_indices, _n_orig, study) = _run_hybrid_pipeline(
                model_type=model_type,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                X_train_raw=X_train_raw, X_test_raw=X_test_raw,
                month_arr=month_arr,
                fpr_threshold=fpr_threshold,
                model_tuning=model_tuning,
                n_trials=n_trials,
                parent_run_id=parent_run.info.run_id,
                device=device,
                cfg=cfg,
            )
            _write_best_params_to_config(
                best_encoder_params, model_type, config_path,
                stage2_params=best_xgb_stage2_params,
                l2_indices=l2_indices,
            )

    logger.info("configs/model_config.yaml updated. Run `make train MODEL=%s`.", model_type)
    return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Tune XGBoost, MLP, GNN, or Transformer+GBDT via Optuna + OOT evaluation."
    )
    parser.add_argument("--trans",   required=True, help="Path to raw transaction CSV")
    parser.add_argument("--id",      required=True, help="Path to raw identity CSV")
    parser.add_argument("--config",  default=None,
                        help="Path to YAML config (default: configs/model_config.yaml)")
    parser.add_argument("--model",   default="xgboost",
                        choices=list(VALID_MODELS),
                        help="Model to tune (default: xgboost)")
    parser.add_argument("--trials",  type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    run_tuning(args.trans, args.id, args.config, args.trials, args.model)
