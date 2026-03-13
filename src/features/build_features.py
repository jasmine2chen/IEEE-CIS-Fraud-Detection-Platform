from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import List

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

def get_feature_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """
    Create a standard scikit-learn pipeline optimized for IEEE-CIS data.
    Implements Frequency Encoding and standard scaling. 
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('freq_encode', FrequencyEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor

def create_magic_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Create Kaggle UID by combining card variables and account start date D1."""
    if 'TransactionDT' in df.columns and 'D1' in df.columns and 'card1' in df.columns and 'addr1' in df.columns:
        df['day'] = df['TransactionDT'] / (24*60*60)
        df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + np.floor(df['day'] - df['D1']).astype(str)
    return df

def normalize_d_columns(df: pd.DataFrame, d_cols: List[str]) -> pd.DataFrame:
    """Convert time-relative D columns to absolute point-in-past measurements."""
    if 'TransactionDT' in df.columns:
        day = df['TransactionDT'] / (24*60*60)
        for col in d_cols:
            if col in df.columns:
                df[f'{col}_normalized'] = day - df[col]
                df = df.drop(columns=[col])
    return df

def add_uid_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Group by magic UID to derive meaningful historical features for the client."""
    if 'uid' not in df.columns:
        return df
        
    # Example aggregations based on Kaggle strategies
    agg_features = {}
    if 'TransactionAmt' in df.columns:
        agg_features['TransactionAmt_uid_mean'] = df.groupby('uid')['TransactionAmt'].transform('mean')
        agg_features['TransactionAmt_uid_std'] = df.groupby('uid')['TransactionAmt'].transform('std')
        
    for d_col in ['D4_normalized', 'D9_normalized', 'D10_normalized', 'D15_normalized']:
        if d_col in df.columns:
            agg_features[f'{d_col}_uid_mean'] = df.groupby('uid')[d_col].transform('mean')
            agg_features[f'{d_col}_uid_std'] = df.groupby('uid')[d_col].transform('std')

    df_agg = pd.DataFrame(agg_features, index=df.index)
    return pd.concat([df, df_agg], axis=1)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply major kaggle feature transformations to the dataframe."""
    df = create_magic_uid(df)
    
    d_cols = [c for c in df.columns if c.startswith('D') and not c.startswith('Device')]
    df = normalize_d_columns(df, d_cols)
    
    df = add_uid_aggregations(df)
    return df
