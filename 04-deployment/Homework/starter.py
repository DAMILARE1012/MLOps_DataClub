#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import sys
import argparse

# Ignore warnings...
import warnings
warnings.filterwarnings("ignore")

def main(year: int, month: int):
    taxi_type = 'yellow'
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    input_file=f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    def read_dataframe(filename):
        df = pd.read_parquet(filename)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
        
        return df

    def prepare_dictionaries(df: pd.DataFrame):
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        dicts = df[categorical].to_dict(orient='records')
        
        return dicts

    def apply_model(input_file, output_file):
        df = read_dataframe(input_file)
        dicts = prepare_dictionaries(df)

        X_val = dv.transform(dicts)
        y_pred = model.predict(X_val)
        
        # What's the standard deviation of the predicted duration for this dataset?
        print("The value of the standard deviation of the predicted duration for this dataset is: ", np.std(y_pred)) 

        df_result = pd.DataFrame()
        df_result['ride_id'] = df['ride_id']
        df_result['predicted_duration'] = y_pred
        
        # ✅ Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

        # Calculate and print mean predicted duration
        mean_duration = np.mean(y_pred)
        print(f"The mean predicted duration is: {mean_duration:.2f}")

        # Print file size
        file_size_bytes = os.path.getsize(output_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

    apply_model(input_file=input_file, output_file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process taxi trip data')
    parser.add_argument('--year', type=int, required=True, help='Year of the data')
    parser.add_argument('--month', type=int, required=True, help='Month of the data')
    
    args = parser.parse_args()
    main(args.year, args.month)



