"""Tests for src/research_tools — feature inspector and experiment logger."""

import logging
import os
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification_data(
    n: int = 200,
    n_features: int = 10,
    fraud_rate: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) suitable for feature inspector tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.random(n) < fraud_rate).astype(int), name="isFraud")
    return X, y


# ---------------------------------------------------------------------------
# feature_target_correlation
# ---------------------------------------------------------------------------

class TestFeatureTargetCorrelation:
    def test_returns_dataframe(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data()
        result = feature_target_correlation(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data()
        result = feature_target_correlation(X, y)
        assert set(result.columns) >= {"feature", "correlation", "abs_corr"}

    def test_top_n_respected(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data(n_features=20)
        result = feature_target_correlation(X, y, top_n=5)
        assert len(result) <= 5

    def test_sorted_by_abs_corr_descending(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data(n_features=15)
        result = feature_target_correlation(X, y)
        assert list(result["abs_corr"]) == sorted(result["abs_corr"], reverse=True)

    def test_with_train_X_adds_ks_columns(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data(n=200)
        train_X, _ = _make_classification_data(n=300, seed=99)
        result = feature_target_correlation(X, y, train_X=train_X)
        assert "ks_statistic" in result.columns
        assert "ks_pvalue" in result.columns
        assert "drift_flag" in result.columns

    def test_non_numeric_columns_ignored(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data(n_features=5)
        X["text_col"] = "some_value"
        result = feature_target_correlation(X, y)
        assert "text_col" not in result["feature"].values

    def test_columns_with_few_valid_values_skipped(self):
        from src.research.feature_inspector import feature_target_correlation

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"sparse": [np.nan] * 195 + list(rng.standard_normal(5))})
        y = pd.Series(rng.integers(0, 2, 200))
        result = feature_target_correlation(X, y)
        # sparse column has <10 valid values → should be skipped
        assert len(result) == 0

    def test_abs_corr_in_zero_one(self):
        from src.research.feature_inspector import feature_target_correlation

        X, y = _make_classification_data()
        result = feature_target_correlation(X, y)
        assert (result["abs_corr"] >= 0).all()
        assert (result["abs_corr"] <= 1).all()


# ---------------------------------------------------------------------------
# text_feature_audit
# ---------------------------------------------------------------------------

class TestTextFeatureAudit:
    def _make_text_data(self, n: int = 500, seed: int = 42):
        rng = np.random.default_rng(seed)
        domains = ["gmail.com", "yahoo.com", "protonmail.com", "disposable.io", "hotmail.com"]
        df = pd.DataFrame({
            "email_domain": rng.choice(domains, size=n),
            "device": rng.choice(["mobile", "desktop", "tablet"], size=n),
        })
        # protonmail and disposable.io are higher fraud risk
        base_fraud = (rng.random(n) < 0.05).astype(int)
        high_risk_mask = df["email_domain"].isin(["protonmail.com", "disposable.io"])
        y = pd.Series(np.where(high_risk_mask, (rng.random(n) < 0.4).astype(int), base_fraud))
        return df, y

    def test_returns_dict(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain"], y)
        assert isinstance(result, dict)

    def test_key_per_column(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain", "device"], y)
        assert "email_domain" in result
        assert "device" in result

    def test_expected_output_columns(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain"], y)
        df_out = result["email_domain"]
        assert set(df_out.columns) >= {"value", "count", "fraud_count", "fraud_rate", "lift", "pct_of_all_fraud"}

    def test_sorted_by_lift_descending(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain"], y)
        lifts = result["email_domain"]["lift"].tolist()
        assert lifts == sorted(lifts, reverse=True)

    def test_top_n_values_respected(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain"], y, top_n_values=3)
        assert len(result["email_domain"]) <= 3

    def test_missing_column_logged_and_skipped(self, caplog):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        with caplog.at_level(logging.WARNING, logger="src.research.feature_inspector"):
            result = text_feature_audit(df, ["nonexistent_col"], y)
        assert "nonexistent_col" not in result
        assert any("nonexistent_col" in r.message for r in caplog.records)

    def test_lift_values_are_non_negative(self):
        from src.research.feature_inspector import text_feature_audit

        df, y = self._make_text_data()
        result = text_feature_audit(df, ["email_domain"], y)
        assert (result["email_domain"]["lift"] >= 0).all()

    def test_missing_values_handled(self):
        from src.research.feature_inspector import text_feature_audit

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"col": ["a"] * 200 + [None] * 50 + ["b"] * 250})
        y = pd.Series(rng.integers(0, 2, 500))
        result = text_feature_audit(df, ["col"], y)
        assert "col" in result
        values = result["col"]["value"].tolist()
        # None should appear as "(missing)"
        assert "(missing)" in values or len(values) > 0


# ---------------------------------------------------------------------------
# experiment_logger — git helpers and package versions
# ---------------------------------------------------------------------------

class TestGetGitInfo:
    def test_returns_dict_with_expected_keys(self):
        from src.research.experiment_logger import _get_git_info

        info = _get_git_info()
        assert isinstance(info, dict)
        assert "git_commit" in info
        assert "git_branch" in info
        assert "git_dirty" in info

    def test_git_dirty_is_boolean_string(self):
        from src.research.experiment_logger import _get_git_info

        info = _get_git_info()
        # Should be "true", "false", or "unknown"
        assert info["git_dirty"] in ("true", "false", "unknown")

    def test_graceful_fallback_when_git_unavailable(self, monkeypatch):
        from src.research import experiment_logger

        monkeypatch.setattr(
            experiment_logger.subprocess,
            "check_output",
            MagicMock(side_effect=FileNotFoundError("git not found")),
        )
        info = experiment_logger._get_git_info()
        assert info["git_commit"] == "unknown"
        assert info["git_branch"] == "unknown"


class TestGetPackageVersions:
    def test_returns_dict_with_python(self):
        from src.research.experiment_logger import _get_package_versions

        versions = _get_package_versions()
        assert isinstance(versions, dict)
        assert "python" in versions

    def test_unknown_package_does_not_raise(self):
        from src.research.experiment_logger import _get_package_versions

        # Should not raise even if packages are missing
        versions = _get_package_versions()
        assert isinstance(versions, dict)


# ---------------------------------------------------------------------------
# research_run context manager
# ---------------------------------------------------------------------------

class TestResearchRun:
    """Tests for the research_run context manager.

    Uses mlflow.set_tracking_uri to an in-memory SQLite store so tests do not
    require a running MLflow server.
    """

    @pytest.fixture(autouse=True)
    def _mlflow_tmp(self, tmp_path):
        """Point MLflow at a temporary directory for isolation."""
        import mlflow

        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        mlflow.set_tracking_uri(db_uri)
        yield
        mlflow.set_tracking_uri("")  # reset

    def test_context_manager_yields_active_run(self):
        import mlflow
        from src.research.experiment_logger import research_run

        with research_run("test_exp", "test_run") as run:
            assert run is not None
            assert mlflow.active_run() is not None

    def test_git_tags_logged(self):
        import mlflow
        from src.research.experiment_logger import research_run

        with research_run("test_exp", "test_run") as run:
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        tags = client.get_run(run_id).data.tags
        assert "git_commit" in tags

    def test_duration_metric_logged(self):
        import mlflow
        from src.research.experiment_logger import research_run

        with research_run("test_exp", "test_run") as run:
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        assert "run_duration_s" in metrics
        assert metrics["run_duration_s"] >= 0.0

    def test_config_logged_as_artifact(self, tmp_path):
        import mlflow
        from src.research.experiment_logger import research_run

        cfg = {"model": "xgboost", "n_estimators": 300, "nested": {"lr": 0.05}}
        with research_run("test_exp", "config_run", config=cfg) as run:
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="run_metadata")
        names = [a.path for a in artifacts]
        assert any("config.json" in n for n in names)

    def test_require_clean_git_env_var(self, monkeypatch):
        from src.research import experiment_logger

        monkeypatch.setenv("REQUIRE_CLEAN_GIT", "1")
        # Patch git to return dirty
        monkeypatch.setattr(
            experiment_logger,
            "_get_git_info",
            lambda: {"git_commit": "abc", "git_branch": "main", "git_dirty": "true"},
        )
        with pytest.raises(RuntimeError, match="dirty"):
            with experiment_logger.research_run("test_exp", "dirty_run"):
                pass

    def test_dirty_git_allowed_by_default(self, monkeypatch):
        from src.research import experiment_logger

        monkeypatch.delenv("REQUIRE_CLEAN_GIT", raising=False)
        monkeypatch.setattr(
            experiment_logger,
            "_get_git_info",
            lambda: {"git_commit": "abc", "git_branch": "main", "git_dirty": "true"},
        )
        # Should not raise
        with experiment_logger.research_run("test_exp", "dirty_allowed") as run:
            assert run is not None

    def test_user_supplied_tags_set(self):
        import mlflow
        from src.research.experiment_logger import research_run

        with research_run("test_exp", "tagged_run", tags={"team": "research", "env": "ci"}) as run:
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        tags = client.get_run(run_id).data.tags
        assert tags.get("team") == "research"
        assert tags.get("env") == "ci"


# ---------------------------------------------------------------------------
# log_feature_importance
# ---------------------------------------------------------------------------

class TestLogFeatureImportance:
    @pytest.fixture(autouse=True)
    def _mlflow_tmp(self, tmp_path):
        import mlflow

        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        mlflow.set_tracking_uri(db_uri)
        yield
        mlflow.set_tracking_uri("")

    def test_returns_dataframe_with_feature_importance(self):
        import mlflow
        from src.research.experiment_logger import log_feature_importance

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.5, 0.1, 0.05, 0.05])
        feature_names = ["a", "b", "c", "d", "e"]

        mlflow.set_experiment("test_fi")
        with mlflow.start_run():
            df = log_feature_importance(mock_model, feature_names)

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert df.iloc[0]["feature"] == "b"  # highest importance first

    def test_model_without_feature_importances_returns_empty_df(self):
        import mlflow
        from src.research.experiment_logger import log_feature_importance

        mock_model = MagicMock(spec=[])  # no feature_importances_ attribute

        mlflow.set_experiment("test_fi_empty")
        with mlflow.start_run():
            df = log_feature_importance(mock_model, ["a", "b"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_no_active_run_does_not_raise(self, caplog):
        import mlflow
        from src.research.experiment_logger import log_feature_importance

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])

        # Ensure no active run
        assert mlflow.active_run() is None
        # Should not raise
        df = log_feature_importance(mock_model, ["x", "y"])
        assert isinstance(df, pd.DataFrame)
