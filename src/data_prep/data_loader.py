import logging

import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to save memory.

    float minimum floor is float32 — float16 has only ~3 decimal digits of
    precision and silently corrupts transaction amounts and temporal features.
    int columns are downcast aggressively since exact integers tolerate it.
    """
    numerics = ["int16", "int32", "int64", "float32", "float64"]
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = str(df[col].dtype)
        if col_type not in numerics:
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type.startswith("int"):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            # float32 is the minimum safe precision for ML features
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Memory reduced from %.2f MB to %.2f MB (%.1f%% reduction)",
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem,
        )
    return df


def load_data(trans_filepath: str, id_filepath: str) -> pd.DataFrame:
    """Load and merge IEEE-CIS dataset from transaction and identity files."""
    df_trans = pd.read_csv(trans_filepath)
    df_id = pd.read_csv(id_filepath)
    df = pd.merge(df_trans, df_id, on="TransactionID", how="left")
    return reduce_mem_usage(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with extreme missing values (>70%)."""
    missing_ratios = df.isna().mean()
    cols_to_keep = df.columns[missing_ratios <= 0.7]
    return df[cols_to_keep]


def prepare_data(trans_filepath: str, id_filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load, clean data and separate features/target."""
    df = load_data(trans_filepath, id_filepath)
    df = clean_data(df)
    if "isFraud" in df.columns:
        return df.drop(columns=["isFraud"]), df["isFraud"]
    return df, pd.Series(dtype=float)
