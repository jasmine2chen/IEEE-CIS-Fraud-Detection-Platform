import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Provides a small dummy dataframe mimicking IEEE data for testing."""
    data = {
        'TransactionID': [1, 2, 3, 4],
        'TransactionDT': [86400, 172800, 259200, 345600],
        'TransactionAmt': [10.0, 500.0, 1500.0, 20.0],
        'card1': [1000, 2000, 3000, 1000],
        'addr1': [315.0, 325.0, 315.0, 315.0],
        'D1': [0.0, 1.0, 2.0, 3.0],
        'isFraud': [0, 0, 1, 0]
    }
    return pd.DataFrame(data)
