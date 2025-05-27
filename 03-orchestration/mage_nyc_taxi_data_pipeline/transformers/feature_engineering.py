# transformers/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.feature_extraction import DictVectorizer
import pickle

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def create_features(datasets: Dict[str, pd.DataFrame], *args, **kwargs) -> Dict:
    """
    Create features for ML training
    """
    print("Starting feature engineering...")
    
    # Prepare categorical and numerical features
    def prepare_features(df):
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        
        # Handle missing values in numerical features
        numerical = ['trip_distance']
        df[numerical] = df[numerical].fillna(0)
        
        return df
    
    # Process both datasets
    processed_datasets = {}
    for name, df in datasets.items():
        processed_datasets[name] = prepare_features(df.copy())
    
    # Create features using DictVectorizer
    def create_feature_matrix(df, dv=None):
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        
        dicts = df[categorical + numerical].to_dict(orient='records')
        
        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
            
        return X, dv
    
    # Create training features
    X_train, dv = create_feature_matrix(processed_datasets['train'])
    y_train = processed_datasets['train']['duration'].values
    
    # Create validation features using same vectorizer
    X_val, _ = create_feature_matrix(processed_datasets['validation'], dv)
    y_val = processed_datasets['validation']['duration'].values
    
    print(f"Feature engineering completed:")
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Validation features shape: {X_val.shape}")
    print(f"  - Feature vocabulary size: {len(dv.vocabulary_)}")
    
    # Return all processed data
    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'vectorizer': dv,
        'train_df': processed_datasets['train'],
        'val_df': processed_datasets['validation']
    }