# Implementation Guide

Complete documentation of the ML Lambda Deployment architecture, components, and data flow.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Inference Pipeline](#inference-pipeline)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Infrastructure](#infrastructure)
7. [Testing Strategy](#testing-strategy)
8. [Configuration](#configuration)

---

## System Overview

ML Lambda Deployment is a complete MLOps pipeline that trains an Iris flower classifier and deploys it as a serverless API on AWS Lambda behind API Gateway.

### Two-Phase Design

| Phase | Environment | Purpose |
|-------|------------|---------|
| **Training** | Local machine | Data processing, model training, serialization |
| **Inference** | AWS Lambda | Load model, validate input, return predictions |

This separation allows training on powerful local hardware while serving predictions through a highly scalable, pay-per-request serverless function.

---

## Component Architecture

```
src/ml_lambda/
├── config.py                 # Centralized configuration (dataclass)
├── lambda_function.py        # Entry point exposing lambda_handler
│
├── data/
│   └── processor.py          # DataProcessor: load Iris, split, normalize
│
├── training/
│   ├── trainer.py            # ModelTrainer: Random Forest + cross-validation
│   └── evaluator.py          # ModelEvaluator: accuracy, F1, confusion matrix
│
├── model/
│   └── serializer.py         # ModelSerializer: save/load with SHA256 integrity
│
├── inference/
│   ├── handler.py            # LambdaHandler: request lifecycle management
│   ├── validator.py          # InputValidator: type/range/size validation
│   └── predictor.py          # Predictor: model inference with probabilities
│
├── deploy/
│   ├── packager.py           # PackageBuilder: ZIP creation (stub)
│   └── deployer.py           # AWSDeployer: Lambda + API Gateway (stub)
│
└── utils/
    ├── logging.py            # StructuredLogger: JSON format for CloudWatch
    └── exceptions.py         # Custom exception hierarchy
```

### Key Design Decisions

- **Single configuration dataclass** (`Config`) holds all parameters in one place
- **Global handler instance** enables Lambda warm start reuse between invocations
- **SHA256 integrity checks** on serialized models prevent corruption
- **Structured JSON logging** enables CloudWatch Insights queries
- **Property-based testing** with Hypothesis catches edge cases

---

## Data Flow

### Training Phase

```
Iris Dataset (150 samples, 4 features, 3 classes)
       │
       ▼
DataProcessor.load_iris()
       │
       ▼
DataProcessor.split_data()  →  80% train / 20% test
       │
       ▼
ModelTrainer.train()  →  RandomForest(n_estimators=100)
       │                   + 5-fold cross-validation
       ▼
ModelEvaluator.evaluate()  →  accuracy, precision, recall, F1
       │
       ▼
ModelSerializer.save()  →  artifacts/model.joblib + SHA256 hash
```

### Inference Phase

```
HTTP POST /predict  →  {"features": [5.1, 3.5, 1.4, 0.2]}
       │
       ▼
API Gateway (CORS, throttling, routing)
       │
       ▼
LambdaHandler.handle()
       │
       ├── _load_model_once()     →  Cold start: load model from disk
       ├── _parse_body()          →  Parse JSON, validate size (max 1KB)
       ├── InputValidator         →  Type check, length check, range warnings
       └── Predictor.predict()    →  Model inference + probabilities
       │
       ▼
Response: {"prediction": 0, "class_name": "setosa",
           "probabilities": [0.95, 0.03, 0.02], "latency_ms": 12.5}
```

---

## Inference Pipeline

### Lambda Handler (`handler.py`)

The `LambdaHandler` class manages the complete request lifecycle:

1. **Cold Start Optimization**: Model loads once per container lifetime via `_load_model_once()`
2. **Body Parsing**: Handles both API Gateway proxy format (string body) and direct invocation (dict body)
3. **Input Validation**: Delegates to `InputValidator` for type, length, and range checks
4. **Prediction**: Delegates to `Predictor` for model inference
5. **Error Handling**: Maps exception types to HTTP status codes (400 for validation, 500 for internal)
6. **CORS Headers**: All responses include `Access-Control-Allow-Origin: *`

### Input Validator (`validator.py`)

Validates incoming API requests:

- **Type checking**: Features must be a list of numbers
- **Length checking**: Exactly 4 features required (Iris dataset)
- **Range warnings**: Logs when values fall outside typical Iris ranges (non-blocking)
- **Body size limit**: Maximum 1KB to prevent abuse
- **Input sanitization**: Strips dangerous characters from string inputs

### Predictor (`predictor.py`)

Wraps scikit-learn model calls:

- Calls `model.predict()` for class label
- Calls `model.predict_proba()` for confidence scores
- Maps numeric prediction to class name (setosa/versicolor/virginica)
- Returns structured `PredictionResult` dataclass

---

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Triggered on pushes to `main`/`develop` and pull requests to `main`:

| Step | Tool | Purpose |
|------|------|---------|
| Lint | ruff | Code style and import checking |
| Type check | mypy | Static type analysis |
| Format | black | Code formatting verification |
| Test | pytest | Unit + property tests with 80% coverage minimum |
| Security | bandit | Static security analysis |
| Package | zip | Build Lambda deployment artifact |

Matrix testing: Python 3.11 and 3.12.

### Continuous Deployment (`.github/workflows/deploy.yml`)

Manual trigger (`workflow_dispatch`) with environment selection:

1. Run tests as pre-deployment gate
2. Train model (ensures latest artifact)
3. Build Lambda ZIP package (code + dependencies + model)
4. Configure AWS credentials from GitHub secrets
5. Upload package to S3
6. Update Lambda function code
7. Smoke test: invoke Lambda with sample input, verify 200 response

Supports `staging` and `production` environments.

---

## Infrastructure

### Terraform Resources (`infrastructure/main.tf`)

| Resource | Purpose |
|----------|---------|
| S3 Bucket | Store Lambda deployment packages (versioned) |
| IAM Role | Lambda execution role with least privilege |
| Lambda Function | Python 3.12, 512MB memory, 30s timeout |
| API Gateway (HTTP) | Routes POST /predict and GET /health |
| CloudWatch Log Groups | 14-day retention for Lambda and API Gateway |
| CloudWatch Alarms | Error rate (>5/min) and duration (>10s) alerts |

### Key Configuration

- **Runtime**: Python 3.12
- **Memory**: 512 MB (sufficient for scikit-learn + model)
- **Timeout**: 30 seconds
- **Handler**: `ml_lambda.lambda_function.lambda_handler`
- **Region**: us-east-1

---

## Testing Strategy

### Unit Tests

Test individual components in isolation:

- Data processing (loading, splitting, normalization)
- Model training (convergence, parameter handling)
- Serialization (save/load roundtrip, integrity checks)
- Validation (valid inputs, edge cases, malformed data)
- Handler (success paths, error paths, cold start)

### Property-Based Tests (Hypothesis)

Verify invariants across randomly generated inputs:

- Validator accepts any list of 4 valid floats
- Validator rejects lists with wrong length
- Validator rejects non-numeric values
- Body size validation catches oversized payloads
- Serializer roundtrip preserves model fidelity

### Coverage Target

Minimum 80% code coverage enforced in CI.

---

## Configuration

All project settings are centralized in `src/ml_lambda/config.py`:

```python
@dataclass
class Config:
    # Model
    version: str = "v1.0.0"
    artifacts_dir: Path = Path("artifacts")

    # Data
    test_size: float = 0.2
    random_state: int = 42

    # Training
    n_estimators: int = 100
    n_cv_folds: int = 5
    accuracy_threshold: float = 0.9

    # Inference
    expected_features: int = 4
    max_body_size: int = 1024  # 1KB

    # AWS
    aws_region: str = "us-east-1"
    lambda_timeout: int = 30
    lambda_memory: int = 256
```

### Exception Hierarchy

```
MLLambdaError (base)
├── DataValidationError      # Invalid training data
├── ModelNotTrainedError     # Model not trained yet
├── ModelNotFoundError       # Model file missing
├── ModelCorruptedError      # Integrity check failed
├── InputValidationError     # Invalid API input
├── PackageTooLargeError     # ZIP exceeds 50MB
├── AWSCredentialsError      # Invalid AWS credentials
└── DeploymentError          # Deployment failure
```

---

## API Reference

### POST /predict

Classify an Iris flower specimen.

**Request:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Features order: sepal_length, sepal_width, petal_length, petal_width (all in cm).

**Success Response (200):**
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [0.95, 0.03, 0.02],
  "latency_ms": 12.5
}
```

**Validation Error (400):**
```json
{
  "errors": ["Se esperan 4 features, recibidos: 3"]
}
```

**Internal Error (500):**
```json
{
  "errors": ["Internal server error"]
}
```

---

## Local Development

```bash
# Install dependencies
poetry install

# Train a model
poetry run train

# Run tests
poetry run pytest tests/ -v --cov=src/ml_lambda

# Lint
poetry run ruff check src/ tests/

# Format
poetry run black src/ tests/

# Type check
poetry run mypy src/ --ignore-missing-imports
```
