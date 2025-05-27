# data_exporters/export_processed_data.py
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Dict

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_processed_data(processed_data: Dict, *args, **kwargs) -> None:
    """
    Export processed data for ML pipeline consumption
    """
    # Create output directory
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    print("Exporting processed data...")
    
    # Save feature matrices and targets
    with open(output_dir / "X_train.pkl", "wb") as f:
        pickle.dump(processed_data['X_train'], f)
    
    with open(output_dir / "X_val.pkl", "wb") as f:
        pickle.dump(processed_data['X_val'], f)
        
    with open(output_dir / "y_train.pkl", "wb") as f:
        pickle.dump(processed_data['y_train'], f)
        
    with open(output_dir / "y_val.pkl", "wb") as f:
        pickle.dump(processed_data['y_val'], f)
        
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(processed_data['vectorizer'], f)
    
    # Save raw dataframes for additional analysis
    processed_data['train_df'].to_parquet(output_dir / "train_df.parquet")
    processed_data['val_df'].to_parquet(output_dir / "val_df.parquet")
    
    print("Data export completed:")
    print(f"  - Files saved to: {output_dir.absolute()}")
    print(f"  - Training samples: {len(processed_data['y_train'])}")
    print(f"  - Validation samples: {len(processed_data['y_val'])}")
    
    # Create a trigger file for Prefect pipeline
    trigger_file = Path("data_ready.txt")
    with open(trigger_file, "w") as f:
        f.write("Data processing completed\n")
        f.write(f"Training samples: {len(processed_data['y_train'])}\n")
        f.write(f"Validation samples: {len(processed_data['y_val'])}\n")
    
    print(f"Trigger file created: {trigger_file.absolute()}")