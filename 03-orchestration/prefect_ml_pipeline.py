#!/usr/bin/env python
import pickle
import mlflow
from pathlib import Path
from typing import Optional
import pandas as pd

from prefect import flow, task, get_logger
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

# Configuration
MODELS_FOLDER = Path("developed_models")
PROCESSED_DATA_DIR = Path("processed_data")
EXPERIMENT_NAME = "nyc-taxi-experiment-hybrid"

MODELS_FOLDER.mkdir(exist_ok=True)

def setup_mlflow():
    tracking_uri = "http://127.0.0.1:5000"  # Or your MLflow URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri

@task(name="Load Processed Data")
def load_processed_data():
    """Load data processed by Mage pipeline"""
    logger = get_logger()
    
    # Check if data is ready
    trigger_file = Path("data_ready.txt")
    if not trigger_file.exists():
        raise FileNotFoundError("Data not ready! Run Mage pipeline first.")
    
    logger.info("Loading processed data from Mage pipeline...")
    
    # Load all processed data
    with open(PROCESSED_DATA_DIR / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    
    with open(PROCESSED_DATA_DIR / "X_val.pkl", "rb") as f:
        X_val = pickle.load(f)
        
    with open(PROCESSED_DATA_DIR / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
        
    with open(PROCESSED_DATA_DIR / "y_val.pkl", "rb") as f:
        y_val = pickle.load(f)
        
    with open(PROCESSED_DATA_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    logger.info(f"Data loaded successfully:")
    logger.info(f"  - Training samples: {len(y_train)}")
    logger.info(f"  - Validation samples: {len(y_val)}")
    logger.info(f"  - Feature dimensions: {X_train.shape[1]}")
    
    return X_train, X_val, y_train, y_val, vectorizer

@task(name="Train ML Model", retries=1, timeout_seconds=3600)
def train_xgb_model(X_train, y_train, X_val, y_val, vectorizer, params: Optional[dict] = None):
    """Train XGBoost model on processed data"""
    logger = get_logger()
    
    default_params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'seed': 42
    }
    
    model_params = params if params else default_params
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("data_source", "mage_pipeline")
        
        # Prepare XGBoost data
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        val_dmatrix = xgb.DMatrix(X_val, label=y_val)
        
        logger.info("Training XGBoost model...")
        
        # Train model
        model = xgb.train(
            params=model_params,
            dtrain=train_dmatrix,
            num_boost_round=100,
            evals=[(val_dmatrix, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred = model.predict(val_dmatrix)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_samples", len(y_train))
        mlflow.log_metric("val_samples", len(y_val))
        
        logger.info(f"Model training completed. RMSE: {rmse:.4f}")
        
        # Save artifacts
        vectorizer_path = MODELS_FOLDER / "vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact(str(vectorizer_path), artifact_path="preprocessor")
        
        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        return run_id, rmse

@flow(
    name="NYC Taxi ML Training Pipeline (Hybrid)",
    description="ML training pipeline that uses data from Mage preprocessing",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def hybrid_ml_pipeline(model_params: Optional[dict] = None):
    """Main ML pipeline that uses Mage-processed data"""
    logger = get_logger()
    
    # Setup MLflow
    tracking_uri = setup_mlflow()
    logger.info(f"Using MLflow: {tracking_uri}")
    
    # Load data processed by Mage
    X_train, X_val, y_train, y_val, vectorizer = load_processed_data()
    
    # Train model
    run_id, rmse = train_xgb_model(X_train, y_train, X_val, y_val, vectorizer, model_params)
    
    logger.info(f"Hybrid pipeline completed! MLflow Run ID: {run_id}, RMSE: {rmse:.4f}")
    return run_id

if __name__ == "__main__":
    # Run the ML pipeline
    result = hybrid_ml_pipeline()
    print(f"Pipeline completed with run ID: {result}")