-- PostgreSQL schema for fraud detection monitoring
-- Tables: drift_metrics, prediction_logs
-- Run automatically via docker-entrypoint-initdb.d when the postgres container starts.

-- ---------------------------------------------------------------------------
-- drift_metrics: one row per drift check run
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                        SERIAL PRIMARY KEY,
    model_type                VARCHAR(50)   NOT NULL,
    model_version             VARCHAR(20),
    evaluation_timestamp      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    window_start              TIMESTAMPTZ,
    window_end                TIMESTAMPTZ,
    n_samples                 INTEGER,
    dataset_drift_detected    BOOLEAN,
    n_drifted_features        INTEGER,
    share_drifted_features    FLOAT,
    prediction_drift_score    FLOAT,
    prediction_drift_detected BOOLEAN,
    mean_fraud_prob           FLOAT,
    fpr_estimate              FLOAT
);

-- ---------------------------------------------------------------------------
-- prediction_logs: one row per scored transaction (for audit / analysis)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                 BIGSERIAL PRIMARY KEY,
    transaction_id     VARCHAR(50),
    model_type         VARCHAR(50)  NOT NULL,
    model_version      VARCHAR(20),
    fraud_probability  FLOAT        NOT NULL,
    is_fraud           BOOLEAN      NOT NULL,
    score_timestamp    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    batch_id           VARCHAR(50)
);

-- ---------------------------------------------------------------------------
-- Indexes for Grafana time-series queries and model-type filtering
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_drift_metrics_evaluation_timestamp
    ON drift_metrics (evaluation_timestamp);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_model_type
    ON drift_metrics (model_type);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_score_timestamp
    ON prediction_logs (score_timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_type
    ON prediction_logs (model_type);
