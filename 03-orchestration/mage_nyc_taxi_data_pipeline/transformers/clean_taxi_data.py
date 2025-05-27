import pandas as pd
from typing import Dict, List, Optional

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def clean_taxi_data(datasets: Dict[str, pd.DataFrame], *args, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Clean and validate taxi data
    """
    cleaned_datasets = {}
    
    for split_name, df in datasets.items():
        print(f"Cleaning {split_name} data...")
        
        # Calculate trip duration
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df['duration'] = df.duration.apply(lambda td: td.total_seconds() / 60)
        
        # Data quality filtering
        initial_count = len(df)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        filtered_count = len(df)
        
        print(f"{split_name}: Filtered {initial_count - filtered_count} invalid records")
        print(f"{split_name}: {filtered_count} records remaining")
        
        # Data quality metrics
        quality_ratio = filtered_count / initial_count
        avg_duration = df.duration.mean()
        
        print(f"{split_name} quality metrics:")
        print(f"  - Data quality ratio: {quality_ratio:.4f}")
        print(f"  - Average duration: {avg_duration:.2f} minutes")
        
        # Store quality metrics for monitoring
        df.attrs[f'{split_name}_quality_metrics'] = {
            'initial_count': initial_count,
            'filtered_count': filtered_count,
            'quality_ratio': quality_ratio,
            'avg_duration': avg_duration
        }
        
        cleaned_datasets[split_name] = df
    
    return cleaned_datasets