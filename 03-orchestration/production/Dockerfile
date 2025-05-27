# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install networking tools for diagnostics
RUN apt-get update && apt-get install -y netcat-traditional iputils-ping && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY nyc_taxi_pipeline.py .
COPY deployment.py .

# Create models directory
RUN mkdir -p models

# Expose port for Prefect server
EXPOSE 4200

# Set environment variables
ENV PREFECT_API_URL="http://localhost:4200/api"
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"

# Run the deployment script when container starts
CMD ["python", "deployment.py"]