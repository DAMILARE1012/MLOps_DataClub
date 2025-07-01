import pytest
import pandas as pd
import os
from datetime import datetime

# 1. Set ENV variables
os.environ["INPUT_FILE_PATTERN"] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = "s3://nyc-duration/in/2023-01.parquet"
options = {
    'client_kwargs': {
        'endpoint_url': os.environ["S3_ENDPOINT_URL"]
    }
}

# 2. Save test data
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# 3. Run the batch prediction script
exit_code = os.system("python batch.py 2023 1")
assert exit_code == 0

# 4. Read result from output
output_file = "s3://nyc-duration/out/2023-01.parquet"
df_output = pd.read_parquet(output_file, storage_options=options)

# 5. Check predicted durations
print("Predicted durations:\n", df_output['predicted_duration'])
print("Sum of durations:", df_output['predicted_duration'].sum())


#```
# predicted mean duration: 18.138625226015364
# Predicted durations:
# 0    23.197149
# 1    13.080101
# Name: predicted_duration, dtype: float64
# Sum of durations: 36.27725045203073

#```