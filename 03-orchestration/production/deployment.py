# deployment.py
from prefect import serve
from nyc_taxi_pipeline import nyc_taxi_training_pipeline
from prefect.client.schemas.schedules import CronSchedule

# Create deployments for different scenarios
if __name__ == "__main__":
    
    # Monthly retraining deployment
    monthly_deployment = nyc_taxi_training_pipeline.to_deployment(
        name="nyc-taxi-monthly-training",
        description="Monthly model retraining for NYC taxi duration prediction",
        schedule=CronSchedule(cron="0 2 1 * *"),  # 2 AM on 1st of each month
        parameters={
            "year": 2024,
            "month": 1,
            "validation_threshold": 12.0
        },
        tags=["ml", "taxi", "production", "monthly"]
    )
    
    # Ad-hoc training deployment
    adhoc_deployment = nyc_taxi_training_pipeline.to_deployment(
        name="nyc-taxi-adhoc-training",
        description="On-demand model training",
        parameters={
            "validation_threshold": 15.0
        },
        tags=["ml", "taxi", "adhoc"]
    )
    
    # Serve deployments
    serve(
        monthly_deployment,
        adhoc_deployment
    )