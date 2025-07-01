import pytest
from datetime import datetime
import pandas as pd
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


# How many rows should be there in the expected dataframe? - 2
def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),          # duration = 9 mins → keep
        (1, 1, dt(1, 2), dt(1, 10)),                # duration = 8 mins → keep
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),       # duration = 0.98 mins → drop (<1 min)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),           # duration = 1441 mins → drop (>60 mins)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    # Expected: only the first two rows should be kept (duration between 1 and 60)
    assert len(actual_df) == 2

    # Optional: check transformation types
    assert actual_df['PULocationID'].dtype == 'object'
    assert actual_df['DOLocationID'].dtype == 'object'
    
    
