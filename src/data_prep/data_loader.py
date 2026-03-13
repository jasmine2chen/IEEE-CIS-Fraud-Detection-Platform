import pandas as pd
import numpy as np
from typing import Tuple

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Iterate through all columns and downcast data types to save memory."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({(100 * (start_mem - end_mem) / start_mem):.1f}% reduction)')
    return df

def load_data(trans_filepath: str, id_filepath: str) -> pd.DataFrame:
    """Load and merge IEEE-CIS dataset from transaction and identity files."""
    df_trans = pd.read_csv(trans_filepath)
    df_id = pd.read_csv(id_filepath)
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
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
    
    if 'isFraud' in df.columns:
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']
        return X, y
    return df, pd.Series()
