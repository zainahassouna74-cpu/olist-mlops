# Olist Late Delivery Prediction – MLOps Project

## Project Overview

This project builds a production-ready machine learning inference service for predicting whether an Olist e-commerce order will be delivered late.

The project started from Jupyter notebooks and was refactored into a structured MLOps repository with:

- Python modules
- Data validation
- DVC data versioning
- MLflow experiment tracking and model registry
- FastAPI inference service
- Automated testing
- Docker and Docker Compose
- CI/CD
- Logging and error handling
- Service monitoring
- Prediction drift detection

The prediction task is binary classification:

- `0` → On-time delivery
- `1` → Late delivery

---

## Project Structure

```text
olist-mlops/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── config/
│   └── config.yaml
│
├── data/
│
├── gx/
│
├── logs/
│
├── models/
│
├── notebooks/
│   ├── 1_read join tables.ipynb
│   ├── 2_create labels.ipynb
│   ├── 3_train val test split.ipynb
│   ├── 4_EDA.ipynb
│   ├── 5_preprocessing.ipynb
│   ├── 6_train tune evaluate.ipynb
│   └── results_summary.json
│
├── requirements/
│   ├── base.txt
│   └── dev.txt
│
├── src/
│   ├── data/
│   │   ├── build_dataset.py
│   │   ├── labels.py
│   │   ├── load.py
│   │   ├── split.py
│   │   └── validate.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── preprocessing.py
│   │
│   ├── inference/
│   │   ├── loader.py
│   │   └── predict.py
│   │
│   ├── monitoring/
│   │   └── drift.py
│   │
│   ├── tracking/
│   │   └── register_model.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data_validation.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_logging_errors.py
│   ├── test_model.py
│   ├── test_monitoring.py
│   ├── test_pipeline_integration.py
│   ├── test_preprocessing.py
│   └── test_schema.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── models.dvc
├── pyproject.toml
└── README.md
```

---

## Requirements

The project uses:

- Python 3.12
- Docker Desktop
- Git
- DVC

Runtime dependencies are stored in:

```text
requirements/base.txt
```

Development dependencies are stored in:

```text
requirements/dev.txt
```

Dependencies are pinned to specific versions to improve reproducibility.

---

## Runtime Dependencies

`requirements/base.txt`:

```text
pandas==2.2.3
numpy==2.3.1
SQLAlchemy==2.0.52
psycopg2-binary==2.9.12
scikit-learn==1.7.0
scipy==1.16.0
joblib==1.5.1
PyYAML==6.0.2
python-dotenv==1.2.3
pydantic==2.13.5
fastapi==0.141.1
uvicorn==0.46.0
mlflow==3.16.1
```

---

## Development Dependencies

`requirements/dev.txt`:

```text
-r base.txt

pytest==9.1.1
black==26.5.1
ruff==0.16.8
dvc==3.67.1
great-expectations==1.23.1
pre-commit==4.6.2
httpx==0.28.1
psycopg2-binary==2.9.12
```

---

## Local Setup

Clone the repository:

```bash
git clone https://github.com/zainahassouna74-cpu/olist-mlops.git
cd olist-mlops
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install development dependencies:

```bash
pip install -r requirements/dev.txt
```

---

## Environment Variables

Create a `.env` file from the example:

```powershell
copy .env.example .env
```

Example `.env.example`:

```env
MLFLOW_DB_USER=postgres
MLFLOW_DB_PASSWORD=your_password_here
MLFLOW_DB_NAME=mlflow
```

The real `.env` file must not be committed to Git.

---

## Configuration

Project configuration is stored in:

```text
config/config.yaml
```

The configuration includes:

- project metadata
- file paths
- model artifact paths
- prediction threshold
- MLflow configuration
- database configuration
- API configuration
- monitoring configuration

Example configuration sections:

```yaml
project:
  name: olist-mlops
  version: "1.0.0"

model:
  model_file: models/logistic_regression_model.pkl
  preprocessor_file: models/preprocessor.pkl
  feature_list_file: models/feature_list.json
  threshold: 0.75

mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: olist_delivery_delay
  registered_model_name: olist_delivery_delay_model
  stage: Production

api:
  host: 0.0.0.0
  port: 8000

monitoring:
  baseline_late_rate: 0.0903
  drift_alert_threshold: 0.10
  min_drift_samples: 20
```

Environment variables can override selected configuration values such as the MLflow tracking URI.

---

## Notebook to Python Module Refactoring

The original notebook workflow was converted into reusable Python modules.

### Data modules

Located in:

```text
src/data/
```

Responsibilities include:

- database access
- dataset construction
- label creation
- time-based train/validation/test splitting
- data validation

### Feature modules

Located in:

```text
src/features/
```

Responsibilities include:

- feature engineering
- preprocessing
- transforming raw API input into model-ready features

### Inference modules

Located in:

```text
src/inference/
```

Responsibilities include:

- loading the registered model
- loading the fitted preprocessor
- loading the saved feature list
- performing inference

No training or fitting is performed inside the inference service.

---

## Feature Engineering

Feature engineering is implemented in:

```text
src/features/build_features.py
```

Generated features include:

- `purchase_year`
- `purchase_month`
- `purchase_dayofweek`
- `purchase_hour`
- `estimated_delivery_days`

The API accepts the original timestamps:

```text
order_purchase_timestamp
order_estimated_delivery_date
```

The production pipeline creates the engineered features automatically.

---

## Preprocessing

Preprocessing is implemented in:

```text
src/features/preprocessing.py
```

The production service uses the fitted preprocessing object created during training.

Inference uses:

```python
preprocessor.transform(...)
```

It does not use:

```python
fit(...)
```

or:

```python
fit_transform(...)
```

This prevents retraining or re-fitting during inference.

---

## Logging and Error Handling

The project uses Python's `logging` library rather than `print` statements in production modules.

Logging includes:

- console logging
- file logging
- INFO-level events
- errors and exceptions
- prediction requests
- model version
- inference latency
- prediction output

Invalid API inputs are rejected using Pydantic validation.

Examples:

- missing required fields → HTTP `422`
- invalid numeric values → HTTP `422`
- invalid dates → HTTP `422`
- unexpected prediction errors → HTTP `500`

Prediction logs are JSON serializable, including datetime values.

---

## Data Versioning with DVC

The project uses DVC to version datasets and model artifacts.

Check the local DVC status:

```bash
dvc status
```

Push artifacts:

```bash
dvc push
```

Pull artifacts:

```bash
dvc pull
```

Check remote synchronization:

```bash
dvc status -c
```

The current development environment uses a configured DVC remote.

### Important Note About the Current DVC Remote

The current DVC remote used during development is a local filesystem remote.

This is sufficient for local development and testing, but a different machine cannot access that local path directly.

For a truly portable clean-machine deployment, the DVC remote should be changed to a shared remote such as:

- S3
- Azure Blob Storage
- Google Cloud Storage
- SSH storage
- another shared storage service

before relying on `dvc pull` from a separate machine.

---

## Data Validation with Great Expectations

Training data is validated using Great Expectations.

Run validation:

```bash
python -m src.data.validate
```

Validation checks include:

- expected schema
- required columns
- missing values
- missing-value rates
- numeric data types
- numeric ranges
- allowed categorical values
- binary target values

Examples include:

- non-negative prices
- non-negative freight values
- valid Brazilian state codes
- valid order status values
- `is_late` restricted to `0` or `1`

If validation fails, the current policy is:

```text
REJECT
```

The pipeline raises an error and does not continue with invalid data.

---

## MLflow Experiment Tracking

MLflow is used for:

- experiment tracking
- model parameters
- project parameters
- evaluation metrics
- preprocessing artifacts
- evaluation artifacts
- trained model artifacts

Register the trained model:

```bash
python -m src.tracking.register_model
```

---

## MLflow Model Registry

The selected model is registered as:

```text
olist_delivery_delay_model
```

The service uses a registered model version and Production stage.

The inference loader retrieves:

- the model
- fitted preprocessor
- feature list
- model version

from MLflow and its artifact store.

The API therefore does not load its prediction model from a notebook folder.

---

## Running Tests

Run the complete test suite with:

```bash
python -m pytest
```

The project currently contains tests for:

- FastAPI endpoints
- request schemas
- invalid input handling
- feature engineering
- preprocessing
- model artifact loading
- model probability output
- inference
- Great Expectations validation behavior
- logging
- error handling
- pipeline integration
- monitoring
- drift detection
- Swagger documentation
- OpenAPI schema

The current test suite contains:

```text
27 tests
```

---

## Code Quality

### Format the code

```bash
black app src tests
```

### Check formatting

```bash
black --check app src tests
```

### Run linting

```bash
ruff check app src tests
```

### Automatically fix supported lint issues

```bash
ruff check app src tests --fix
```

---

## FastAPI Service

Run the API locally:

```bash
python -m uvicorn app.main:app --reload
```

The service is available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

OpenAPI schema:

```text
http://localhost:8000/openapi.json
```

---

## API Routes

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

### Model Information

```http
GET /model-info
```

Returns:

- model name
- model version
- machine learning task

Example structure:

```json
{
  "model_name": "logistic_regression",
  "model_version": "3",
  "task": "late_delivery_classification"
}
```

The exact model version may change when a new model version is registered.

---

### Single Prediction

```http
POST /predict
```

Example request:

```json
{
  "customer_zip_code_prefix": 13023,
  "customer_city": "campinas",
  "customer_state": "SP",
  "item_count": 1,
  "total_price": 120.0,
  "total_freight": 18.0,
  "payment_count": 1,
  "total_payment": 138.0,
  "max_installments": 3,
  "unique_products": 1,
  "unique_sellers": 1,
  "unique_categories": 1,
  "order_purchase_timestamp": "2018-05-10T14:00:00",
  "order_estimated_delivery_date": "2018-05-30T14:00:00"
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.12,
  "model_version": "3"
}
```

The probability and model version shown above are examples and may differ between model versions and requests.

---

### Batch Prediction

```http
POST /batch-predict
```

The batch endpoint accepts a list of valid orders and returns:

- number of predictions
- predictions for each order
- model version

---

## Automatic API Documentation

Swagger UI is automatically generated by FastAPI.

Open:

```text
http://localhost:8000/docs
```

Available routes include:

- `/health`
- `/metrics`
- `/monitoring/drift`
- `/predict`
- `/batch-predict`
- `/model-info`

The OpenAPI specification is available at:

```text
http://localhost:8000/openapi.json
```

---

## Service Monitoring

Monitoring metrics are exposed through:

```http
GET /metrics
```

The endpoint reports:

- request count
- prediction count
- error count
- error rate
- average latency
- late prediction count
- on-time prediction count
- late prediction rate
- model version

Example structure:

```json
{
  "request_count": 5,
  "prediction_count": 5,
  "error_count": 0,
  "error_rate": 0.0,
  "average_latency_ms": 8.25,
  "prediction_distribution": {
    "late": 1,
    "on_time": 4,
    "late_rate": 0.2
  },
  "model_version": "3"
}
```

---

## Prediction Logging

Predictions are stored in:

```text
logs/predictions.jsonl
```

Each prediction record stores:

- timestamp
- request input
- predicted class
- probability
- model version
- latency
- placeholder for the future actual delivery status

Example field:

```json
{
  "actual_delivery_status": null
}
```

This allows predictions to be evaluated later when real delivery outcomes become available.

---

## Drift Monitoring

Prediction drift is checked through:

```http
GET /monitoring/drift
```

The monitoring logic compares the current predicted late-delivery rate with the baseline late-delivery rate.

Current configuration:

```yaml
baseline_late_rate: 0.0903
drift_alert_threshold: 0.10
min_drift_samples: 20
```

The current alert policy is:

> Trigger a drift alert when the absolute difference between the current late-prediction rate and the baseline late rate is at least 10 percentage points after at least 20 predictions have been collected.

Possible statuses include:

```text
no_data
insufficient_data
ok
alert
```

---

## Docker Image

Build the API image:

```bash
docker build -t olist-mlops-api .
```

The Docker image uses:

```text
python:3.12-slim
```

The image contains only the runtime project components required by the service and model initialization workflow.

The full notebooks are not copied into the image.

---

## Docker Compose

Docker Compose runs the complete service stack.

Services include:

- PostgreSQL database
- MLflow server
- MLflow artifact storage
- model initialization container
- FastAPI service

Start the complete stack:

```bash
docker compose up --build
```

Stop the stack:

```bash
docker compose down
```

Check status:

```bash
docker compose ps
```

Expected long-running services:

```text
db
mlflow
api
```

The `model-init` service runs once, registers the model into MLflow, and exits successfully.

---

## Docker Service URLs

FastAPI:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

MLflow:

```text
http://localhost:5000
```

Health endpoint:

```text
http://localhost:8000/health
```

A successful health response is:

```json
{
  "status": "ok"
}
```

---

## Docker and MLflow Networking

When running locally outside Docker, MLflow may use the local tracking configuration.

Inside Docker Compose, the environment variable:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

overrides the local MLflow URI.

This allows:

```text
model-init
```

and:

```text
api
```

to communicate with the MLflow service through the Docker Compose network.

---

## CI/CD

GitHub Actions configuration is stored in:

```text
.github/workflows/ci.yml
```

The workflow runs on:

- every push
- every pull request

The pipeline performs:

1. repository checkout
2. Python setup
3. development dependency installation
4. Black format check
5. Ruff lint check
6. automated tests
7. Docker image build
8. GitHub Container Registry login
9. Docker image tagging
10. Docker image push

A failing step stops the pipeline automatically.

The CI workflow has been verified successfully on GitHub Actions.

---

## GitHub Container Registry

After a successful push workflow, the Docker image is published to GitHub Container Registry.

The CI pipeline uses:

```text
ghcr.io
```

and authenticates using:

```text
GITHUB_TOKEN
```

provided by GitHub Actions.

No registry password is stored directly in the repository.

---

## Development Validation Workflow

Before committing changes, the following checks can be run:

```bash
black --check app src tests
ruff check app src tests
python -m pytest
```

To automatically format and fix supported lint issues:

```bash
black app src tests
ruff check app src tests --fix
```

---

## Clean Machine Workflow

A typical setup workflow is:

```bash
git clone https://github.com/zainahassouna74-cpu/olist-mlops.git
cd olist-mlops

copy .env.example .env

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements/dev.txt

python -m pytest

docker compose up --build
```

### DVC Note

If model or data artifacts must be restored through DVC, run:

```bash
dvc pull
```

However, the currently configured development DVC remote is local to the original development machine.

For a different clean machine to use `dvc pull`, the DVC remote must first be configured to a shared accessible storage backend.

---

## Production Inference Flow

```text
Raw API Request
        ↓
Pydantic Request Validation
        ↓
Feature Engineering
        ↓
Saved Fitted Preprocessor
        ↓
MLflow Registered Model
        ↓
Prediction Probability
        ↓
Configured Threshold
        ↓
Final Prediction
        ↓
Prediction Logging
        ↓
Monitoring Metrics
        ↓
Drift Monitoring
```

No model training occurs inside the inference service.

No preprocessing fitting occurs inside the inference service.

---

## Prediction Task

The machine learning task is:

```text
Late Delivery Classification
```

The target is:

```text
is_late
```

Target meaning:

```text
0 = On-time delivery
1 = Late delivery
```

---

## Definition of Done

The project provides:

- a structured MLOps repository
- configuration management
- separated runtime and development dependencies
- pinned dependencies
- notebook logic refactored into Python modules
- reusable feature engineering
- reusable preprocessing
- fitted artifact reuse during inference
- logging and error handling
- DVC data and artifact versioning
- Great Expectations validation
- MLflow experiment tracking
- MLflow model registry
- model versioning
- automated unit tests
- data tests
- model tests
- integration tests
- FastAPI inference service
- single-order prediction
- batch prediction
- automatic Swagger documentation
- Docker image
- Docker Compose stack
- PostgreSQL
- MLflow artifact storage
- GitHub Actions CI/CD
- Docker image publishing
- request monitoring
- latency monitoring
- error monitoring
- prediction distribution monitoring
- prediction logging
- prediction drift detection

---

## Project Status

The production-style inference pipeline and its supporting MLOps components are implemented and tested.

The main application workflow supports:

```text
request
→ validation
→ feature engineering
→ preprocessing
→ inference
→ response
→ logging
→ monitoring
```

The project is designed to demonstrate the process of moving a machine learning workflow from notebooks into a structured, tested, containerized inference service.