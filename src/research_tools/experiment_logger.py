"""Experiment logging utilities — thin wrappers over MLflow for research reproducibility.

The biggest reproducibility failure in research contexts is running experiments from a
dirty working tree and later being unable to identify which code version produced a
result.  ``research_run()`` automatically captures git state, package versions, and
wall-clock duration for every experiment, with zero boilerplate at the call site.

Usage
-----
    from src.research_tools.experiment_logger import research_run, log_feature_importance

    with research_run("fraud_ablation", "no_uid_aggs", config=cfg) as run:
        mlflow.log_metric("roc_auc", 0.95)
        log_feature_importance(model, feature_names)
"""

import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _get_git_info() -> Dict[str, str]:
    """Return git commit hash, branch, and dirty status as a dict.

    Falls back gracefully if git is unavailable or the directory is not a repo.
    """
    info: Dict[str, str] = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty":  "unknown",
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_dirty"] = "true" if dirty_output else "false"
    except Exception:
        pass
    return info


def _get_package_versions() -> Dict[str, str]:
    """Return versions of key ML packages for reproducibility."""
    packages = ["torch", "xgboost", "sklearn", "numpy", "pandas", "mlflow", "optuna"]
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


# ---------------------------------------------------------------------------
# research_run context manager
# ---------------------------------------------------------------------------

@contextmanager
def research_run(
    experiment_name: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    config: Optional[dict] = None,
    tracking_uri: Optional[str] = None,
    require_clean_git: Optional[bool] = None,
) -> Generator[Any, None, None]:
    """Context manager for reproducible research experiments.

    Wraps ``mlflow.start_run()`` and automatically logs:
      - Git commit hash, branch, and dirty status
      - Python version and key package versions (torch, xgboost, sklearn, ...)
      - Full config dict as a JSON artifact (if provided)
      - Wall-clock run duration as ``run_duration_s`` metric on exit

    Args:
        experiment_name:   MLflow experiment name (created if it doesn't exist).
        run_name:          Human-readable run name for the MLflow UI.
        tags:              Optional additional MLflow tags to set.
        config:            Config dict to log as a JSON artifact.
        tracking_uri:      MLflow tracking server URI (uses current default if None).
        require_clean_git: If True, raise RuntimeError when working tree is dirty.
                           Defaults to the REQUIRE_CLEAN_GIT env var (False if unset).

    Yields:
        The active ``mlflow.ActiveRun`` object.

    Example
    -------
        with research_run("fraud_experiments", "xgboost_no_uid_aggs") as run:
            mlflow.log_metric("roc_auc", 0.95)
            print("run_id:", run.info.run_id)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Git hygiene check
    git_info = _get_git_info()
    _require_clean = require_clean_git
    if _require_clean is None:
        _require_clean = os.getenv("REQUIRE_CLEAN_GIT", "0") == "1"
    if _require_clean and git_info.get("git_dirty") == "true":
        raise RuntimeError(
            "Working tree is dirty. Commit or stash changes before running experiments, "
            "or unset REQUIRE_CLEAN_GIT to allow dirty runs."
        )
    if git_info.get("git_dirty") == "true":
        logger.warning(
            "Working tree is dirty (uncommitted changes). "
            "Set REQUIRE_CLEAN_GIT=1 to block dirty runs."
        )

    pkg_versions = _get_package_versions()

    mlflow.set_experiment(experiment_name)
    t_start = time.perf_counter()

    with mlflow.start_run(run_name=run_name) as run:
        # Git metadata
        mlflow.set_tags(git_info)
        # Package versions
        mlflow.set_tags({f"pkg_{k}": v for k, v in pkg_versions.items()})
        # Platform
        mlflow.set_tag("platform", platform.platform())
        # User-supplied tags
        if tags:
            mlflow.set_tags(tags)

        # Log config as artifact
        if config is not None:
            try:
                import json, tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    cfg_path = os.path.join(tmp, "config.json")
                    with open(cfg_path, "w") as f:
                        json.dump(config, f, indent=2, default=str)
                    mlflow.log_artifact(cfg_path, artifact_path="run_metadata")
            except Exception as exc:
                logger.warning("Failed to log config artifact: %s", exc)

        try:
            yield run
        finally:
            duration = time.perf_counter() - t_start
            mlflow.log_metric("run_duration_s", round(duration, 2))
            logger.info("Run '%s' completed in %.1fs (run_id=%s)",
                        run_name, duration, run.info.run_id)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def log_dataframe_stats(
    df: pd.DataFrame,
    name: str,
    sample_n: int = 5,
) -> None:
    """Log DataFrame shape, dtypes, null counts, and a sample to MLflow.

    Useful for confirming the exact data split used in each experiment —
    a common source of undetected errors in research code.

    Args:
        df:       DataFrame to summarise.
        name:     Artifact name prefix (used for the JSON filename).
        sample_n: Number of rows to include in the sample.
    """
    if mlflow.active_run() is None:
        logger.warning("log_dataframe_stats: no active MLflow run — skipping.")
        return

    stats = {
        "shape":      list(df.shape),
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes":      df.dtypes.astype(str).to_dict(),
        "sample":      df.head(sample_n).to_dict(orient="records"),
    }
    try:
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{name}_stats.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            mlflow.log_artifact(path, artifact_path="data_stats")
    except Exception as exc:
        logger.warning("log_dataframe_stats failed: %s", exc)


def log_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 30,
    artifact_name: str = "feature_importance",
) -> pd.DataFrame:
    """Log top-N feature importances as a CSV artifact.

    Supports any model with a ``feature_importances_`` attribute
    (XGBoost, RandomForest, etc.).

    Args:
        model:         Fitted model with ``feature_importances_``.
        feature_names: Feature names aligned with model input columns.
        top_n:         Number of top features to log (by importance).
        artifact_name: Artifact subdirectory name in the MLflow run.

    Returns:
        Full importance DataFrame sorted by importance descending.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        logger.warning("Model has no feature_importances_ — skipping.")
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature":    feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if mlflow.active_run() is not None:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = os.path.join(tmp, f"{artifact_name}.csv")
                df.head(top_n).to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path=artifact_name)
        except Exception as exc:
            logger.warning("log_feature_importance failed: %s", exc)

    return df
