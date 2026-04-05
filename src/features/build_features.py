import bisect
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Temporal D-columns in IEEE-CIS dataset (explicit list prevents accidentally
# normalizing non-temporal columns that happen to start with 'D')
_D_TEMPORAL_COLS = [f"D{i}" for i in range(1, 16)]

# Columns produced by feature engineering that must be dropped before the
# ColumnTransformer sees the data (not model inputs).
_ENGINEERED_COLS_TO_DROP = ["TransactionID", "TransactionDT", "uid", "day", "Month"]

# Column groups for uid aggregations
_C_COLS_FOR_AGG  = ["C1", "C9", "C11", "C13"]
_M_COLS_FOR_AGG  = ["M1", "M4", "M5", "M7", "M8", "M9"]  # binary T/F — encoded to 1/0
_D_NORM_COLS_FOR_AGG = [                                   # after normalize_d_columns()
    "D2_normalized", "D4_normalized", "D9_normalized",
    "D10_normalized", "D15_normalized",
]
# (source_column, output_feature_name)
_CONSISTENCY_COLS = [
    ("P_emaildomain", "uid_unique_email_count"),
    ("DeviceInfo",    "uid_unique_DeviceInfo_count"),
]


# ---------------------------------------------------------------------------
# uid aggregation helpers
# ---------------------------------------------------------------------------

def _expanding_nunique_shifted(series: pd.Series) -> pd.Series:
    """O(N) count of distinct non-null values seen strictly before each row.

    Equivalent to expanding().nunique().shift(1) but O(N) via a running set
    instead of recomputing the unique set from scratch each step.
    """
    result = np.zeros(len(series), dtype=np.float32)
    seen: set = set()
    for i, val in enumerate(series):
        result[i] = len(seen)
        if val == val and val is not None:  # fast notna check
            seen.add(val)
    return pd.Series(result, index=series.index)


def _expanding_percentile_shifted(series: pd.Series) -> pd.Series:
    """O(N log N) percentile rank of each value within its prior history.

    Uses a sorted list + binary search so each insertion and rank query
    is O(log k) where k is the number of prior values.
    Returns NaN when there is no prior history (first transaction).
    """
    n = len(series)
    result = np.full(n, np.nan, dtype=np.float32)
    sorted_hist: list = []
    for i, val in enumerate(series):
        if sorted_hist and val == val:
            result[i] = bisect.bisect_left(sorted_hist, val) / len(sorted_hist)
        if val == val and val is not None:
            bisect.insort(sorted_hist, val)
    return pd.Series(result, index=series.index)


def _count_in_time_window(times: pd.Series, window_seconds: int) -> pd.Series:
    """O(N) count of prior transactions within a sliding time window.

    For transaction i at time T, counts how many earlier transactions j < i
    satisfy T - window_seconds < times[j] <= T (strictly before current).
    Assumes `times` is already sorted ascending — guaranteed because
    add_uid_aggregations() sorts the full DataFrame before applying this.
    """
    arr = times.to_numpy()
    n = len(arr)
    result = np.zeros(n, dtype=np.int32)
    left = 0
    for i in range(n):
        while left < i and arr[i] - arr[left] > window_seconds:
            left += 1
        result[i] = i - left   # excludes current transaction
    return pd.Series(result, index=times.index)


# ---------------------------------------------------------------------------
# sklearn components
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features by their frequency in the training set."""
    def __init__(self):
        self.frequencies = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.frequencies[col] = X[col].value_counts(dropna=False).to_dict()
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(X).copy()
        for col in X_encoded.columns:
            if col in self.frequencies:
                X_encoded[col] = X_encoded[col].map(self.frequencies[col]).fillna(0)
        return X_encoded.values


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer wrapping all domain feature engineering.

    Baking feature engineering into the Pipeline as a step ensures the exact
    same transformations are applied at training time and inference time,
    eliminating training/serving skew. The fitted Pipeline object (including
    this step) is serialized to disk and loaded by the API.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = build_features(pd.DataFrame(X).copy())
        drop_cols = [c for c in _ENGINEERED_COLS_TO_DROP if c in X.columns]
        return X.drop(columns=drop_cols)


def get_full_pipeline() -> Pipeline:
    """Build a complete, serializable sklearn Pipeline.

    Stages:
      1. FraudFeatureEngineer — UID creation, D-column normalization, UID aggs
      2. ColumnTransformer    — median imputation + scaling (numeric),
                                frequency encoding + scaling (categorical)

    Both numeric and categorical paths exit as ~N(0,1) so that all columns
    arrive at the same scale when passed to gradient-based models (MLP,
    TabTransformer, GNN). Without scaling the categorical path, raw frequency
    counts (e.g. 47823) would dwarf normalised numerics (~[-3, 3]), causing
    the FeatureTokenizer gradients to be dominated by high-frequency categories.
    XGBoost is unaffected (splits are scale-invariant) so this change is a
    no-op for tree-only training.

    This is the single artifact that must be fit at training time and
    loaded at inference time. Callers pass raw DataFrames; the pipeline
    handles all feature engineering and preprocessing internally.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer",     SimpleImputer(strategy="constant", fill_value="missing")),
        ("freq_encode", FrequencyEncoder()),
        ("scaler",      StandardScaler()),  # normalise raw counts to ~N(0,1)
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude=["object", "category"])),
            ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "category"])),
        ]
    )
    return Pipeline(steps=[
        ("feature_engineering", FraudFeatureEngineer()),
        ("preprocessing", preprocessor),
    ])


# ---------------------------------------------------------------------------
# Feature engineering helpers (called by FraudFeatureEngineer.transform)
# ---------------------------------------------------------------------------

def create_magic_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Create Kaggle UID by combining card variables and account start date D1."""
    required = {"TransactionDT", "D1", "card1", "addr1"}
    if required.issubset(df.columns):
        df["day"] = df["TransactionDT"] / (24 * 60 * 60)
        df["uid"] = (
            df["card1"].astype(str) + "_"
            + df["addr1"].astype(str) + "_"
            + np.floor(df["day"] - df["D1"]).astype(str)
        )
    return df


def normalize_d_columns(df: pd.DataFrame, d_cols: List[str]) -> pd.DataFrame:
    """Convert time-relative D columns to absolute point-in-past measurements."""
    if "TransactionDT" not in df.columns:
        return df
    day = df["TransactionDT"] / (24 * 60 * 60)
    for col in d_cols:
        if col in df.columns:
            df[f"{col}_normalized"] = day - df[col]
            df = df.drop(columns=[col])
    return df


def add_uid_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical uid features with no look-ahead leakage.

    Every feature uses an expanding window shifted by 1 row so each
    transaction only sees events that occurred strictly before it.
    First transactions for a uid receive NaN (→ median-imputed downstream).

    Feature groups
    --------------
    Expanding stats   — TransactionAmt, C/D-normalized/M columns
    Count             — uid_transaction_count (prior txns for this uid)
    Time              — days_since_first_seen, time_since_last_txn,
                        avg_time_between_txns
    Velocity          — txn_count in last 1 h / 24 h (O(N) sliding window)
    Consistency       — unique email & DeviceInfo counts, email consistency flag
    Deviation         — amount deviation/ratio/outlier/percentile vs uid history

    Serving alignment: the expanding window mirrors a production lookup store
    (e.g. Redis) pre-populated from training history. New uids at serve time
    return NaN → training-median fallback, exactly matching this behaviour.

    GNN note: this function is also called for graph construction (to get uid
    and TransactionDT for edges). The extra features computed here are unused
    by the GNN — only uid/TransactionDT are extracted from X_eng DataFrames.
    """
    if "uid" not in df.columns or "TransactionDT" not in df.columns:
        return df

    # Sort by time so expanding() and velocity windows accumulate correctly.
    # Restore original row order at the end so index alignment is preserved.
    original_index = df.index
    df = df.sort_values("TransactionDT")

    agg: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 1. TransactionAmt — mean, std (basis for deviation features later)
    # ------------------------------------------------------------------
    if "TransactionAmt" in df.columns:
        grp = df.groupby("uid")["TransactionAmt"]
        amt_mean = grp.transform(lambda x: x.expanding().mean().shift(1))
        amt_std  = grp.transform(lambda x: x.expanding(min_periods=2).std().shift(1))
        agg["TransactionAmt_uid_mean"] = amt_mean
        agg["TransactionAmt_uid_std"]  = amt_std

    # ------------------------------------------------------------------
    # 2. C columns — mean only (ratio/velocity signals)
    # ------------------------------------------------------------------
    for col in _C_COLS_FOR_AGG:
        if col in df.columns:
            agg[f"{col}_uid_mean"] = df.groupby("uid")[col].transform(
                lambda x: x.expanding().mean().shift(1)
            )

    # ------------------------------------------------------------------
    # 3. D-normalized columns — mean + std (temporal pattern signals)
    # ------------------------------------------------------------------
    for col in _D_NORM_COLS_FOR_AGG:
        if col in df.columns:
            grp = df.groupby("uid")[col]
            agg[f"{col}_uid_mean"] = grp.transform(
                lambda x: x.expanding().mean().shift(1)
            )
            agg[f"{col}_uid_std"] = grp.transform(
                lambda x: x.expanding(min_periods=2).std().shift(1)
            )

    # ------------------------------------------------------------------
    # 4. M columns — binary-encoded (T=1, F=0), then mean / M8 std
    #    Mean = fraction of prior txns where M-flag was True.
    # ------------------------------------------------------------------
    for col in _M_COLS_FOR_AGG:
        if col in df.columns:
            m_num = df[col].map({"T": 1.0, "F": 0.0})
            agg[f"{col}_uid_mean"] = m_num.groupby(df["uid"]).transform(
                lambda x: x.expanding().mean().shift(1)
            )
    if "M8" in df.columns:
        m8_num = df["M8"].map({"T": 1.0, "F": 0.0})
        agg["M8_uid_std"] = m8_num.groupby(df["uid"]).transform(
            lambda x: x.expanding(min_periods=2).std().shift(1)
        )

    # ------------------------------------------------------------------
    # 5. Count features
    # ------------------------------------------------------------------
    # cumcount() = 0, 1, 2 … for each uid — already backward-looking (no shift needed)
    agg["uid_transaction_count"] = df.groupby("uid").cumcount().astype(float)

    # ------------------------------------------------------------------
    # 6. Time features
    # ------------------------------------------------------------------
    if "day" in df.columns:
        # Days since the uid's first observed transaction (0 for first txn).
        # No shift needed — min(day) is always in the past.
        agg["uid_days_since_first_seen"] = df.groupby("uid")["day"].transform(
            lambda x: x - x.min()
        )

    dt_grp = df.groupby("uid")["TransactionDT"]
    # diff() is naturally backward-looking: T[i] - T[i-1], NaN for first txn.
    agg["uid_time_since_last_txn"]   = dt_grp.transform(lambda x: x.diff())
    agg["uid_avg_time_between_txns"] = dt_grp.transform(
        lambda x: x.diff().expanding().mean().shift(1)
    )

    # ------------------------------------------------------------------
    # 7. Velocity features (O(N) sliding window per uid group)
    # ------------------------------------------------------------------
    agg["uid_txn_count_1h"]  = dt_grp.transform(
        lambda x: _count_in_time_window(x, 3_600)
    )
    agg["uid_txn_count_24h"] = dt_grp.transform(
        lambda x: _count_in_time_window(x, 86_400)
    )

    # ------------------------------------------------------------------
    # 8. Consistency features (O(N) running-set per uid group)
    # ------------------------------------------------------------------
    for src_col, feat_name in _CONSISTENCY_COLS:
        if src_col in df.columns:
            agg[feat_name] = df.groupby("uid")[src_col].transform(
                _expanding_nunique_shifted
            )

    if "uid_unique_email_count" in agg:
        # 0 prior emails (new uid) → consistent by default (1.0)
        agg["uid_is_email_consistent"] = (agg["uid_unique_email_count"] <= 1).astype(float)

    # Concatenate all aggregation columns in one shot (avoids N DataFrame copies)
    df = pd.concat([df, pd.DataFrame(agg, index=df.index)], axis=1)

    # ------------------------------------------------------------------
    # 9. Deviation features — derived from already-computed uid stats
    #    All NaN-safe: NaN uid_mean/std → NaN deviation (imputed downstream)
    # ------------------------------------------------------------------
    if "TransactionAmt" in df.columns and "TransactionAmt_uid_mean" in df.columns:
        deviation = df["TransactionAmt"] - df["TransactionAmt_uid_mean"]
        df["TransactionAmt_deviation_from_uid_mean"] = deviation
        df["TransactionAmt_ratio_to_uid_mean"] = (
            df["TransactionAmt"] / df["TransactionAmt_uid_mean"].replace(0.0, np.nan)
        )
        if "TransactionAmt_uid_std" in df.columns:
            # NaN when std is NaN (first txn or single-txn uid) — imputed downstream
            df["TransactionAmt_is_uid_outlier"] = (
                deviation.abs() > 2.0 * df["TransactionAmt_uid_std"]
            ).astype(float).where(df["TransactionAmt_uid_std"].notna(), np.nan)

        # O(N log N) percentile via sorted insertion (see _expanding_percentile_shifted)
        df["TransactionAmt_percentile_in_uid"] = (
            df.groupby("uid")["TransactionAmt"].transform(_expanding_percentile_shifted)
        )

    logger.debug(
        "add_uid_aggregations: added %d uid features to %d rows",
        len(df.columns) - len(original_index),   # rough count
        len(df),
    )

    return df.loc[original_index]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all domain feature engineering transformations.

    Returns the enriched DataFrame including intermediate columns (uid, day).
    FraudFeatureEngineer calls this and then drops non-model columns before
    handing off to the ColumnTransformer.
    """
    df = create_magic_uid(df)
    df = normalize_d_columns(df, _D_TEMPORAL_COLS)
    df = add_uid_aggregations(df)
    return df
