import pandas as pd
from src.preprocessing.data_loader import clean_data
from src.feature_engineering.build_features import build_features

def test_clean_data(sample_data):
    """Test data cleaning removes extreme NaNs."""
    import numpy as np
    n = len(sample_data)
    sample_data['extreme_missing'] = [np.nan] * (n - 1) + [1.0]
    clean_df = clean_data(sample_data)
    assert 'extreme_missing' not in clean_df.columns
    assert 'TransactionAmt' in clean_df.columns

def test_magic_uid_generation(sample_data):
    """Test Magic UID is generated accurately."""
    df_engineered = build_features(sample_data)
    assert 'day' in df_engineered.columns
    assert 'uid' in df_engineered.columns
    # Check if transaction 1 and 4 have the same UID (same card1, addr1, and start day)
    assert df_engineered.loc[0, 'uid'] == df_engineered.loc[3, 'uid']
