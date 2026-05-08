"""Unified training entry point — supports XGBoost, MLP+XGBoost, Transformer+XGBoost, GNN+XGBoost.

Model selection
---------------
Pass --model xgboost (default) | mlp_xgboost | transformer_xgboost | gnn_xgboost.
The active model type is also logged as an MLflow tag.

All models:
- Use the same OOT split (months 0-5 train, month 6 test).
- Use the same feature pipeline (fitted once, serialised to joblib).
- Use FPR-based early stopping at serving.fraud_threshold_prob.
- Log params, metrics, and artifacts to MLflow.

Neural hybrids (mlp_xgboost, transformer_xgboost, gnn_xgboost):
  Two-stage — encoder pre-training then XGBoost on enriched [orig || embedding] features.

Usage
-----
    make train                                        # XGBoost
    make train MODEL=mlp_xgboost                      # MLP→XGBoost
    make train MODEL=transformer_xgboost              # TabTransformer→XGBoost
    make train MODEL=gnn_xgboost                      # GraphSAGE→XGBoost
    make tune-then-train                              # tune → update YAML → retrain
"""

import argparse
import logging
import os
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.preprocessing.data_loader import prepare_data
from src.evaluation.metrics import auc_at_max_fpr, fpr_sweep, log_fpr_sweep
from src.feature_engineering.build_features import build_features, get_full_pipeline
from src.training.models.mlp_tree import extract_mlp_embeddings, train_mlp_xgboost
from src.training.models.transformer_tree import extract_transformer_embeddings, train_transformer_xgboost
from src.training.models.gnn_tree import GNNArtifact, extract_gnn_embeddings, train_gnn_xgboost
from src.training.models.tree_models import get_xgboost_model
from src.deployment.registry import register_model, promote_to_champion, CANONICAL_XGB_ARTIFACT

logger = logging.getLogger(__name__)

VALID_MODELS = ("xgboost", "mlp_xgboost", "transformer_xgboost", "gnn_xgboost")

# Neural hybrid model types (two-stage encoder + XGBoost)
_NEURAL_HYBRIDS = ("mlp_xgboost", "transformer_xgboost", "gnn_xgboost")


def time_consistency_split(
    df: pd.DataFrame,
    train_months: int = 6,
):
    """Return train/test indices using an OOT split.

    Split layout (default: months 0-5 train, final month test):
      - Train: months [0, train_months)  — all months before the test month
      - Test:  final month in the dataset — out-of-time evaluation

    Does NOT mutate the input DataFrame.
    """
    SECONDS_IN_MONTH = 2_592_000  # 30 days
    month     = np.floor(df["TransactionDT"] / SECONDS_IN_MONTH)
    train_idx = df.index[month < train_months]
    test_idx  = df.index[month == month.max()]

    logger.info(
        "OOT split — train: %d samples (months 0–%d) | test: %d samples (month %d)",
        len(train_idx), train_months - 1,
        len(test_idx), int(month.max()),
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Shared post-training evaluation
# ---------------------------------------------------------------------------

def _log_fraud_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    max_fpr: float = 0.05,
) -> None:
    """Run FPR sweep and log fraud-specific metrics to MLflow and logger."""
    sweep = fpr_sweep(y_true, y_prob, amounts=amounts)
    log_fpr_sweep(sweep)

    partial_auc = auc_at_max_fpr(y_true, y_prob, max_fpr=max_fpr)
    mlflow.log_metric("auc_at_5pct_fpr", partial_auc)
    logger.info("AUC@5%%FPR: %.4f", partial_auc)

    for target_pct, key in [(1.0, "1pct"), (2.0, "2pct"), (5.0, "5pct")]:
        row = min(sweep, key=lambda r: abs(r["actual_fpr_pct"] - target_pct))
        mlflow.log_metric(f"recall_at_{key}_fpr", row["recall"])
        if amounts is not None:
            mlflow.log_metric(f"dollar_recall_at_{key}_fpr", row["dollar_recall"])


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model-specific training helpers
# ---------------------------------------------------------------------------

def _train_xgboost(cfg, X_train, y_train, X_test, y_test, amounts=None) -> None:
    """Train XGBoost with FPR-based early stopping and log all artifacts."""
    xgb_params        = cfg["xgboost_params"]
    training_cfg      = cfg["training"]
    fpr_threshold     = cfg["serving"]["fraud_threshold_prob"]
    early_stopping_rounds = training_cfg.get("early_stopping_rounds", 50)

    selected = cfg.get("xgboost_selected_features")
    if selected:
        feature_idx = np.array(selected, dtype=int)
        X_train = X_train[:, feature_idx]
        X_test  = X_test[:, feature_idx]
        logger.info("Feature selection applied: %d / %d features retained",
                    len(feature_idx), feature_idx.max() + 1)
        mlflow.log_param("n_selected_features", len(feature_idx))

    mlflow.log_params(xgb_params)
    mlflow.set_tag("model_type", "xgboost")
    mlflow.log_param("fpr_threshold", fpr_threshold)

    logger.info(
        "Training XGBoost on %d samples, %d features — FPR early stopping @ %.2f",
        X_train.shape[0], X_train.shape[1], fpr_threshold,
    )
    model = get_xgboost_model(
        params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
        fpr_threshold=fpr_threshold,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    best_iter = getattr(model, "best_iteration", xgb_params.get("n_estimators"))
    logger.info("Stopped at iteration %d (early_stopping_rounds=%d)",
                best_iter, early_stopping_rounds)

    preds = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    logger.info("OOT AUC: %.4f", auc)

    mlflow.log_metric("OOT_AUC", auc)
    mlflow.log_metric("best_iteration", best_iter)
    y_v = y_test.values if hasattr(y_test, "values") else y_test
    _log_fraud_metrics(y_v, preds, amounts=amounts)

    os.makedirs("models", exist_ok=True)
    model.set_params(eval_metric=None)  # closure can't be pickled; metric only needed during training
    joblib.dump(model, "models/xgboost_fraud_model.joblib")
    mlflow.xgboost.log_model(model, artifact_path=CANONICAL_XGB_ARTIFACT["xgboost"])
    logger.info("XGBoost model saved.")


def _train_mlp_xgboost(cfg, X_train, y_train, X_test, y_test, amounts=None) -> None:
    """Train MLPEncoder + XGBoost hybrid.

    If tune.py has run the hybrid pipeline, cfg carries:
        mlp_xgboost_params      → encoder architecture params (Phase A best)
        mlp_xgboost_stage2_params → XGBoost stage-2 params (Phase B best)
        mlp_xgboost_selected    → L2 RFE indices into [orig || embed] matrix

    When those keys are present uses the tuned path; otherwise falls back to
    the original integrated two-stage flow.
    """
    encoder_params = cfg.get("mlp_xgboost_params", {})
    stage2_params  = cfg.get("mlp_xgboost_stage2_params")
    l2_selected    = cfg.get("mlp_xgboost_selected")
    fpr_threshold  = cfg["serving"]["fraud_threshold_prob"]
    device         = _detect_device()

    mlflow.log_params({k: v for k, v in encoder_params.items()})
    mlflow.set_tag("model_type", "mlp_xgboost")
    mlflow.log_param("fpr_threshold", fpr_threshold)
    mlflow.log_param("device", device)

    logger.info(
        "Training MLP+XGBoost on %d samples, %d features  device=%s",
        X_train.shape[0], X_train.shape[1], device,
    )

    if stage2_params and l2_selected is not None:
        # Tuned path: two explicit stages with L2 feature selection
        mlflow.log_params({f"stage2_{k}": v for k, v in stage2_params.items()})
        mlflow.log_param("n_l2_selected", len(l2_selected))

        combined_for_enc = {**stage2_params, **encoder_params}
        encoder, _ = train_mlp_xgboost(
            X_train, y_train, X_test, y_test,
            params=combined_for_enc, fpr_threshold=fpr_threshold,
            device=device, save_path="models/mlp_xgboost",
        )

        embed_train     = extract_mlp_embeddings(encoder, X_train, device=device)
        embed_test      = extract_mlp_embeddings(encoder, X_test,  device=device)
        X_comb_train    = np.concatenate([X_train, embed_train], axis=1)
        X_comb_test     = np.concatenate([X_test,  embed_test],  axis=1)

        feat_idx        = np.array(l2_selected, dtype=int)
        X_sel_train     = X_comb_train[:, feat_idx]
        X_sel_test      = X_comb_test[:,  feat_idx]

        early_stopping  = stage2_params.get("xgb_early_stopping_rounds",
                                             cfg["training"].get("early_stopping_rounds", 50))
        xgb_model = get_xgboost_model(
            params=stage2_params, early_stopping_rounds=early_stopping,
            fpr_threshold=fpr_threshold,
        )
        xgb_model.fit(X_sel_train, y_train,
                      eval_set=[(X_sel_test, y_test)], verbose=False)

        preds = xgb_model.predict_proba(X_sel_test)[:, 1]
        os.makedirs("models/mlp_xgboost", exist_ok=True)
        xgb_model.set_params(eval_metric=None)
        joblib.dump(xgb_model, "models/mlp_xgboost/xgboost_stage2.joblib")
        mlflow.xgboost.log_model(xgb_model, artifact_path="mlp_xgboost_stage2_model")

    else:
        encoder, xgb_model = train_mlp_xgboost(
            X_train, y_train, X_test, y_test,
            params=encoder_params, fpr_threshold=fpr_threshold,
            device=device, save_path="models/mlp_xgboost",
        )
        embed_test = extract_mlp_embeddings(encoder, X_test, device=device)
        X_sel_test = np.concatenate([X_test, embed_test], axis=1)
        preds      = xgb_model.predict_proba(X_sel_test)[:, 1]

    y_v = y_test.values if hasattr(y_test, "values") else y_test
    auc = roc_auc_score(y_v, preds)
    logger.info("MLP+XGBoost OOT AUC: %.4f", auc)
    mlflow.log_metric("OOT_AUC", auc)
    _log_fraud_metrics(y_v, preds, amounts=amounts)
    mlflow.xgboost.log_model(xgb_model, artifact_path=CANONICAL_XGB_ARTIFACT["mlp_xgboost"])


def _train_transformer_xgboost(cfg, X_train, y_train, X_test, y_test, amounts=None) -> None:
    """Train TabTransformerEncoder + XGBoost hybrid.

    Same config-driven tuned/untuned pattern as MLP+XGBoost.
    Tuning keys: transformer_xgboost_params, transformer_xgboost_stage2_params,
                 transformer_xgboost_selected.
    """
    encoder_params = cfg.get("transformer_xgboost_params", {})
    stage2_params  = cfg.get("transformer_xgboost_stage2_params")
    l2_selected    = cfg.get("transformer_xgboost_selected")
    fpr_threshold  = cfg["serving"]["fraud_threshold_prob"]
    device         = _detect_device()

    mlflow.log_params({k: v for k, v in encoder_params.items()})
    mlflow.set_tag("model_type", "transformer_xgboost")
    mlflow.log_param("fpr_threshold", fpr_threshold)
    mlflow.log_param("device", device)

    logger.info(
        "Training Transformer+XGBoost on %d samples, %d features  device=%s",
        X_train.shape[0], X_train.shape[1], device,
    )

    if stage2_params and l2_selected is not None:
        mlflow.log_params({f"stage2_{k}": v for k, v in stage2_params.items()})
        mlflow.log_param("n_l2_selected", len(l2_selected))

        combined = {**stage2_params, **encoder_params}
        encoder, _ = train_transformer_xgboost(
            X_train, y_train, X_test, y_test,
            params=combined, fpr_threshold=fpr_threshold,
            device=device, save_path="models/transformer_xgboost",
        )

        embed_train  = extract_transformer_embeddings(encoder, X_train, device=device)
        embed_test   = extract_transformer_embeddings(encoder, X_test,  device=device)
        X_comb_train = np.concatenate([X_train, embed_train], axis=1)
        X_comb_test  = np.concatenate([X_test,  embed_test],  axis=1)

        feat_idx    = np.array(l2_selected, dtype=int)
        X_sel_train = X_comb_train[:, feat_idx]
        X_sel_test  = X_comb_test[:,  feat_idx]

        es = stage2_params.get("xgb_early_stopping_rounds",
                               cfg["training"].get("early_stopping_rounds", 50))
        xgb_model = get_xgboost_model(
            params=stage2_params, early_stopping_rounds=es, fpr_threshold=fpr_threshold,
        )
        xgb_model.fit(X_sel_train, y_train,
                      eval_set=[(X_sel_test, y_test)], verbose=False)
        preds = xgb_model.predict_proba(X_sel_test)[:, 1]

        os.makedirs("models/transformer_xgboost", exist_ok=True)
        xgb_model.set_params(eval_metric=None)
        joblib.dump(xgb_model, "models/transformer_xgboost/xgboost_stage2.joblib")
        mlflow.xgboost.log_model(xgb_model, artifact_path="transformer_xgboost_stage2_model")

    else:
        encoder, xgb_model = train_transformer_xgboost(
            X_train, y_train, X_test, y_test,
            params=encoder_params, fpr_threshold=fpr_threshold,
            device=device, save_path="models/transformer_xgboost",
        )
        embed_test = extract_transformer_embeddings(encoder, X_test, device=device)
        X_sel_test = np.concatenate([X_test, embed_test], axis=1)
        preds      = xgb_model.predict_proba(X_sel_test)[:, 1]

    y_v = y_test.values if hasattr(y_test, "values") else y_test
    auc = roc_auc_score(y_v, preds)
    logger.info("Transformer+XGBoost OOT AUC: %.4f", auc)
    mlflow.log_metric("OOT_AUC", auc)
    _log_fraud_metrics(y_v, preds, amounts=amounts)
    mlflow.xgboost.log_model(xgb_model, artifact_path=CANONICAL_XGB_ARTIFACT["transformer_xgboost"])


def _train_gnn_xgboost(
    cfg,
    X_train, y_train, X_test, y_test,
    card1_train=None, card1_test=None,
    amounts=None,
) -> None:
    """Train GraphSAGEEncoder + XGBoost hybrid.

    Tuning keys: gnn_xgboost_params, gnn_xgboost_stage2_params,
                 gnn_xgboost_selected.
    card1_train / card1_test are required for neighbourhood aggregation.
    """
    encoder_params = cfg.get("gnn_xgboost_params", {})
    stage2_params  = cfg.get("gnn_xgboost_stage2_params")
    l2_selected    = cfg.get("gnn_xgboost_selected")
    fpr_threshold  = cfg["serving"]["fraud_threshold_prob"]
    device         = "cpu"  # GNN training is CPU-based (full-batch sparse ops)

    mlflow.log_params({k: v for k, v in encoder_params.items()})
    mlflow.set_tag("model_type", "gnn_xgboost")
    mlflow.log_param("fpr_threshold", fpr_threshold)
    mlflow.log_param("device", device)

    logger.info(
        "Training GNN+XGBoost on %d samples, %d features  device=%s",
        X_train.shape[0], X_train.shape[1], device,
    )

    if stage2_params and l2_selected is not None:
        mlflow.log_params({f"stage2_{k}": v for k, v in stage2_params.items()})
        mlflow.log_param("n_l2_selected", len(l2_selected))

        combined = {**stage2_params, **encoder_params}
        artifact, _ = train_gnn_xgboost(
            X_train, y_train, X_test, y_test,
            card1_train=card1_train, card1_test=card1_test,
            params=combined, fpr_threshold=fpr_threshold,
            device=device, save_path="models/gnn_xgboost",
        )

        embed_train  = extract_gnn_embeddings(artifact, X_train, card1_train, device=device)
        embed_test   = extract_gnn_embeddings(artifact, X_test,  card1_test,  device=device)
        X_comb_train = np.concatenate([X_train, embed_train], axis=1)
        X_comb_test  = np.concatenate([X_test,  embed_test],  axis=1)

        feat_idx    = np.array(l2_selected, dtype=int)
        X_sel_train = X_comb_train[:, feat_idx]
        X_sel_test  = X_comb_test[:,  feat_idx]

        es = stage2_params.get("xgb_early_stopping_rounds",
                               cfg["training"].get("early_stopping_rounds", 50))
        xgb_model = get_xgboost_model(
            params=stage2_params, early_stopping_rounds=es, fpr_threshold=fpr_threshold,
        )
        xgb_model.fit(X_sel_train, y_train,
                      eval_set=[(X_sel_test, y_test)], verbose=False)
        preds = xgb_model.predict_proba(X_sel_test)[:, 1]

        os.makedirs("models/gnn_xgboost", exist_ok=True)
        xgb_model.set_params(eval_metric=None)
        joblib.dump(xgb_model, "models/gnn_xgboost/xgboost_stage2.joblib")
        mlflow.xgboost.log_model(xgb_model, artifact_path="gnn_xgboost_stage2_model")

    else:
        artifact, xgb_model = train_gnn_xgboost(
            X_train, y_train, X_test, y_test,
            card1_train=card1_train, card1_test=card1_test,
            params=encoder_params, fpr_threshold=fpr_threshold,
            device=device, save_path="models/gnn_xgboost",
        )
        embed_test = extract_gnn_embeddings(artifact, X_test, card1_test, device=device)
        X_sel_test = np.concatenate([X_test, embed_test], axis=1)
        preds      = xgb_model.predict_proba(X_sel_test)[:, 1]

    y_v = y_test.values if hasattr(y_test, "values") else y_test
    auc = roc_auc_score(y_v, preds)
    logger.info("GNN+XGBoost OOT AUC: %.4f", auc)
    mlflow.log_metric("OOT_AUC", auc)
    _log_fraud_metrics(y_v, preds, amounts=amounts)
    mlflow.xgboost.log_model(xgb_model, artifact_path=CANONICAL_XGB_ARTIFACT["gnn_xgboost"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(
    trans_path: str,
    id_path: str,
    config_path: Optional[str] = None,
    model_type: Optional[str] = None,
) -> None:
    """Main training loop — model_type overrides config model.type if provided."""
    cfg          = load_config(config_path)
    training_cfg = cfg["training"]

    active_model = model_type or cfg["model"]["type"]
    if active_model not in VALID_MODELS:
        raise ValueError(
            f"model.type must be one of {VALID_MODELS}, got '{active_model}'"
        )

    mlflow.set_tracking_uri(training_cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(training_cfg["mlflow_experiment_name"])

    with mlflow.start_run() as run:
        logger.info("Loading and preparing data...")
        X, y = prepare_data(trans_path, id_path)

        train_idx, test_idx    = time_consistency_split(X)
        X_train_raw, y_train   = X.loc[train_idx], y.loc[train_idx]
        X_test_raw,  y_test    = X.loc[test_idx],  y.loc[test_idx]

        logger.info("Fitting feature pipeline (feature engineering + preprocessing)...")
        full_pipeline = get_full_pipeline()
        X_train_proc  = full_pipeline.fit_transform(X_train_raw)
        X_test_proc   = full_pipeline.transform(X_test_raw)

        amounts = (
            X_test_raw["TransactionAmt"].values
            if "TransactionAmt" in X_test_raw.columns else None
        )

        # card1 values (raw, pre-pipeline) needed for GNN neighbourhood construction
        card1_train = (
            X_train_raw["card1"].values if "card1" in X_train_raw.columns else None
        )
        card1_test = (
            X_test_raw["card1"].values if "card1" in X_test_raw.columns else None
        )

        if active_model == "xgboost":
            _train_xgboost(cfg, X_train_proc, y_train, X_test_proc, y_test, amounts=amounts)
        elif active_model == "mlp_xgboost":
            _train_mlp_xgboost(cfg, X_train_proc, y_train, X_test_proc, y_test, amounts=amounts)
        elif active_model == "transformer_xgboost":
            _train_transformer_xgboost(cfg, X_train_proc, y_train, X_test_proc, y_test, amounts=amounts)
        else:  # gnn_xgboost
            _train_gnn_xgboost(
                cfg, X_train_proc, y_train, X_test_proc, y_test,
                card1_train=card1_train, card1_test=card1_test,
                amounts=amounts,
            )

        # Register model to MLflow Model Registry
        try:
            version = register_model(
                run_id=run.info.run_id,
                model_type=active_model,
                tracking_uri=training_cfg["mlflow_tracking_uri"],
            )
            mlflow.set_tag("registry_version", version)
            logger.info("Model registered: %s version %s", active_model, version)
        except Exception as exc:
            logger.warning("Model registration failed (non-fatal): %s", exc)

        # Feature pipeline shared by all models — always saved
        os.makedirs("models", exist_ok=True)
        joblib.dump(full_pipeline, "models/feature_pipeline.joblib")
        mlflow.sklearn.log_model(full_pipeline, artifact_path="feature_pipeline")
        logger.info("Feature pipeline saved to models/ and logged to MLflow.")

        # Save reference dataset for drift monitoring.
        try:
            import tempfile
            ref_sample = X_test_raw.sample(
                min(50_000, len(X_test_raw)), random_state=42
            ).reset_index(drop=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                ref_path = os.path.join(tmp_dir, "reference.parquet")
                ref_sample.to_parquet(ref_path, index=False)
                mlflow.log_artifact(ref_path, artifact_path="reference_stats")
            logger.info(
                "Reference stats saved (%d rows) to MLflow artifact store.",
                len(ref_sample),
            )
        except Exception as exc:
            logger.warning("Failed to save reference stats (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--trans",  required=True, help="Path to raw transaction CSV")
    parser.add_argument("--id",     required=True, help="Path to raw identity CSV")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (default: configs/model_config.yaml)")
    parser.add_argument("--model",  default=None, choices=list(VALID_MODELS),
                        help="Override model.type from config")
    args = parser.parse_args()
    train(args.trans, args.id, args.config, args.model)
