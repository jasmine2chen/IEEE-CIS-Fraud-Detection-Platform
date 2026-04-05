import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Dummy DataFrame mimicking IEEE-CIS structure for unit tests.

    Design constraints:
    - card1=1000 / addr1=315 appears 4 times so UID aggregations (mean, std)
      are computed over a group of size 4 — avoiding NaN std values.
    - card1=2000 / addr1=325 appears 2 times for a second distinct UID.
    - Covers both legitimate (isFraud=0) and fraudulent (isFraud=1) rows.
    - D1 values vary so D1_normalized differs across rows.
    """
    data = {
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "TransactionDT": [86400, 172800, 259200, 345600, 432000, 518400],
        "TransactionAmt": [10.0, 500.0, 1500.0, 20.0, 300.0, 75.0],
        "card1": [1000, 2000, 1000, 1000, 1000, 2000],
        "addr1": [315.0, 325.0, 315.0, 315.0, 315.0, 325.0],
        "D1": [0.0, 1.0, 2.0, 3.0, 1.0, 2.0],
        "isFraud": [0, 0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)
