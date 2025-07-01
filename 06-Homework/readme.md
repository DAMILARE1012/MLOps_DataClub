# 🧪 MLOps Zoomcamp Homework – Unit & Integration Testing with Pytest and Localstack

This project demonstrates writing **unit** and **integration tests** for a batch data pipeline using `pytest`, mocking S3 with `localstack`, and improving code testability through refactoring.

---

## 📦 Project Overview

We work with NYC yellow taxi trip data to:
- Predict ride durations using a pickled model (`model.bin`)
- Write predictions to S3 (mocked with Localstack)
- Use `pytest` to write unit and integration tests
- Parameterize paths using environment variables

---

## 🧰 Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlops-zoomcamp-homework.git
cd mlops-zoomcamp-homework


### 2 - Install dependencies with Pipenv
pip install pipenv

# Create virtual environment using specific Python version (e.g., 3.10)
pipenv --python "C:/Path/To/Python310/python.exe"

# Activate environment
pipenv shell

# Install dependencies
pipenv install pandas scikit-learn pyarrow
pipenv install --dev pytest

```

## 🧪 Questions & Answers

### ✅ Q1: Refactoring
#### What should the if block look like in batch.py after refactoring?

```
if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)

```

### ✅ Q2: Installing Pytest
#### What is the second file to create inside the tests/ folder ?
    - ✅ __init__.py – This allows test_batch.py to import batch.py.

## ✅ Q3: Unit Test – Row Count
#### Given this mock data:

```
data = [
    (None, None, dt(1, 1), dt(1, 10)),        # ~9 mins (invalid IDs)
    (1, 1, dt(1, 2), dt(1, 10)),              # 8 mins ✅
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),     # 59 mins ✅
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),         # > 24 hrs ❌
]

```
How many rows remain after filtering?
✅ Answer: 2 rows

### ✅ Q4: AWS CLI Option
##### Which option connects AWS CLI to Localstack? - ✅ --endpoint-url

### ✅ Q5: Docker Compose Setup
```
# docker-compose.yaml
version: "3.3"

services:
  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3
      - DEBUG=1
    volumes:
      - "./localstack:/tmp/localstack"
      - "/var/run/docker.sock:/var/run/docker.sock"
```

#### Start with: docker-compose up -d
#### The size of the data - 3620

### Q6: Input File Size in S3
#### We save the mock DataFrame to:
```
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)
```
#### The sum of the prediction - 26.28
