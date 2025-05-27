# data_loaders/taxi_data_loader.py

import pandas as pd
from typing import Dict, List, Optional
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
    
@data_loader
def load_taxi_data(*args, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load NYC taxi data for training and validation
    """
    # Get parameters (you can set these in Mage UI)
    year = kwargs.get('year', 2021)
    train_month = kwargs.get('train_month', 1)
    val_month = train_month + 1 if train_month < 12 else 1
    val_year = year if train_month < 12 else year + 1
    
    datasets = {}
    
    # Load training data
    train_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{train_month:02d}.parquet'
    print(f"Loading training data from: {train_url}")
    datasets['train'] = pd.read_parquet(train_url)
    print(f"Training data loaded: {len(datasets['train'])} records")
    
    # Load validation data  
    val_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{val_year}-{val_month:02d}.parquet'
    print(f"Loading validation data from: {val_url}")
    datasets['validation'] = pd.read_parquet(val_url)
    print(f"Validation data loaded: {len(datasets['validation'])} records")
    
    return datasets