FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install pandas==1.4.2 numpy==1.22.4 scikit-learn==1.0.2 pyarrow==8.0.0

ENTRYPOINT ["python", "starter.py"]

