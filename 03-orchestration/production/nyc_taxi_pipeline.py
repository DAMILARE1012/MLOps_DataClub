#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import mlflow
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb


import prefect
from prefect import flow, task
from prefect.logging import get_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.system import Secret
from prefect.artifacts import create_table_artifact


# Configuration
MODELS_FOLDER = Path("models")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "nyc-taxi-experiment"

# Setup
MODELS_FOLDER.mkdir(exist_ok=True)

def setup_mlflow():
    """Setup MLflow connection and experiment"""
    # Use environment variable if set, otherwise use default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri

@task(name="Data Ingestion", retries=3, retry_delay_seconds=60)
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """
    Ingest NYC taxi data from S3 with robust error handling
    """
    logger = get_logger()
    
    try:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        logger.info(f"Downloading data from {url}")
        
        df = pd.read_parquet(url)
        logger.info(f"Successfully loaded {len(df)} records")
        
        # Data processing
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        
        # Data quality checks
        initial_count = len(df)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        filtered_count = len(df)
        
        logger.info(f"Filtered {initial_count - filtered_count} records with invalid duration")
        
        if filtered_count == 0:
            raise ValueError("No valid records after filtering!")
        
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        
        # Log data quality metrics
        quality_metrics = {
            "total_records": initial_count,
            "valid_records": filtered_count,
            "data_quality_ratio": filtered_count / initial_count,
            "avg_duration": df.duration.mean(),
            "unique_routes": df['PU_DO'].nunique()
        }
        
        logger.info(f"Data quality metrics: {quality_metrics}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data for {year}-{month:02d}: {str(e)}")
        raise

@task(name="Feature Engineering", retries=2)
def create_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple:
    """
    Create features with proper validation
    """
    logger = get_logger()
    
    def create_X(df, dv=None):
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        
        # Handle missing values
        df = df.copy()
        df[numerical] = df[numerical].fillna(0)
        
        dicts = df[categorical + numerical].to_dict(orient='records')

        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
            
        return X, dv
    
    try:
        X_train, dv = create_X(df_train)
        X_val, _ = create_X(df_val, dv)
        
        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values
        
        # Feature validation
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Validation features shape: {X_val.shape}")
        logger.info(f"Feature vocabulary size: {len(dv.vocabulary_)}")
        
        return X_train, X_val, y_train, y_val, dv
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

@task(name="Model Training", retries=1, timeout_seconds=3600)
def train_model(X_train, y_train, X_val, y_val, dv, params: Optional[dict] = None) -> str:
    """
    Train XGBoost model with MLflow tracking
    """
    logger = get_logger()
    
    # Default parameters
    default_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }
    
    model_params = params if params else default_params
    
    with mlflow.start_run() as run:
        try:
            # Log parameters
            mlflow.log_params(model_params)
            
            # Prepare data
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)
            
            logger.info("Starting model training...")
            
            # Train model
            booster = xgb.train(
                params=model_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Evaluate model
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("train_samples", len(y_train))
            mlflow.log_metric("val_samples", len(y_val))
            
            logger.info(f"Model RMSE: {rmse:.4f}")
            
            # Save artifacts
            preprocessor_path = MODELS_FOLDER / "preprocessor.b"
            with open(preprocessor_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
            
            # Log model
            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
            
            # Save run metadata
            run_id = run.info.run_id
            run_id_path = MODELS_FOLDER / "run_id.txt"
            with open(run_id_path, "w") as f:
                f.write(run_id)
            
            # Create Prefect artifact for model metrics
            create_table_artifact(
                key="model-performance",
                table=[
                    {"metric": "RMSE", "value": rmse},
                    {"metric": "Training Samples", "value": len(y_train)},
                    {"metric": "Validation Samples", "value": len(y_val)},
                    {"metric": "MLflow Run ID", "value": run_id}
                ],
                description="Model training results"
            )
            
            logger.info(f"Model training completed successfully. Run ID: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

@task(name="Model Validation")
def validate_model(run_id: str, min_rmse_threshold: float = 15.0) -> bool:
    """
    Validate model performance against thresholds
    """
    logger = get_logger()
    
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        rmse = run.data.metrics.get('rmse')
        
        if rmse is None:
            raise ValueError("RMSE metric not found in run")
        
        is_valid = rmse < min_rmse_threshold
        
        logger.info(f"Model validation: RMSE={rmse:.4f}, Threshold={min_rmse_threshold}, Valid={is_valid}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

@flow(
    name="NYC Taxi Duration Prediction Pipeline",
    description="End-to-end pipeline for training taxi duration prediction models",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def nyc_taxi_training_pipeline(
    year: int = 2021,
    month: int = 1,
    model_params: Optional[dict] = None,
    validation_threshold: float = 15.0
) -> str:
    """
    Main pipeline flow
    """
    logger = get_logger()
    
    # Setup MLflow connection (respects environment variables)
    tracking_uri = setup_mlflow()
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
    
    logger.info(f"Starting pipeline for {year}-{month:02d}")
    
    # Calculate next month for validation
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    
    # Data ingestion (can run in parallel)
    train_data_future = read_dataframe.submit(year, month)
    val_data_future = read_dataframe.submit(next_year, next_month)
    
    # Wait for both data loading tasks
    train_data = train_data_future.result()
    val_data = val_data_future.result()
    
    # Feature engineering
    X_train, X_val, y_train, y_val, dv = create_features(train_data, val_data)
    
    # Model training
    run_id = train_model(X_train, y_train, X_val, y_val, dv, model_params)
    
    # Model validation
    is_valid = validate_model(run_id, validation_threshold)
    
    if not is_valid:
        logger.warning(f"Model did not meet validation criteria!")
        # In production, you might want to trigger retraining or alerts
    
    logger.info(f"Pipeline completed successfully! MLflow Run ID: {run_id}")
    return run_id

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model on NYC taxi data with Prefect")
    parser.add_argument("--year", type=int, default=2021, required=True, help="Year of the data to train on")
    parser.add_argument("--month", type=int, default=1, required=True, help="Month of the data to train on")
    parser.add_argument("--threshold", type=float, default=15.0, help="RMSE validation threshold")
    
    args = parser.parse_args()
    
    # Run the pipeline
    result = nyc_taxi_training_pipeline(
        year=args.year,
        month=args.month,
        validation_threshold=args.threshold
    )
    
    print(f"Pipeline completed with run ID: {result}")