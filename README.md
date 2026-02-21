# ML Lambda Deployment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [Español](README.es.md)

Deploy machine learning models to AWS Lambda with API Gateway. A complete MLOps pipeline from training to serverless inference.

## Overview

This project implements an end-to-end ML deployment workflow:

- **Data Processing**: Load, validate, split, and normalize the Iris dataset
- **Model Training**: Train a Random Forest classifier with cross-validation and comprehensive metrics
- **Model Serialization**: Save trained models with metadata and integrity verification
- **Input Validation**: Robust validation with type checking, range warnings, and sanitization
- **Serverless Inference**: Deploy as an AWS Lambda function behind API Gateway (in progress)
- **Structured Logging**: JSON-formatted logs for observability
- **Property-Based Testing**: Comprehensive test coverage with Hypothesis

## Current Status

**Completed Components:**

- Project setup with Poetry
- Data processing pipeline with validation
- Model training with cross-validation and metrics
- Model serialization with SHA256 integrity checks
- API input validation (type checking, size limits, sanitization)
- Structured JSON logging system
- Comprehensive test suite (unit + property tests with Hypothesis)

**In Progress:**

- Lambda handler implementation
- AWS deployment automation
- API Gateway configuration

**Pending:**

- Integration tests for end-to-end flow
- Deployment packaging and scripts
- Production deployment guide

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOCAL PIPELINE                           │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  Dataset │───▶│ Process  │───▶│  Train   │───▶│ Serialize│  │
│   │   Iris   │    │  & Split │    │  Model   │    │  Model   │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AWS DEPLOYMENT                           │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  Client  │───▶│   API    │───▶│  Lambda  │───▶│  Model   │  │
│   │   HTTP   │    │ Gateway  │    │ Handler  │    │ Predict  │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- AWS CLI configured (for deployment)

## Installation

```bash
git clone https://github.com/JoseJulianMosqueraFuli/ml-lambda-deployment.git
cd ml-lambda-deployment
poetry install
```

## Usage

### Train a Model

```bash
poetry run train
```

This will:

1. Load and preprocess the Iris dataset (80/20 train/test split)
2. Train a Random Forest classifier with cross-validation
3. Evaluate metrics (accuracy, precision, recall, F1-score, confusion matrix)
4. Save the model with metadata to `artifacts/`

**Example output:**
```
Training model...
Cross-validation scores: [0.95, 0.97, 0.96, 0.94, 0.98]
Mean CV accuracy: 0.96 (+/- 0.03)
Test accuracy: 0.97
Model saved to: artifacts/iris_model_v1.0.0.joblib
```

### Validate Input

The validator ensures API inputs are safe and well-formed:

```python
from ml_lambda.inference.validator import InputValidator

# Valid input
features = [5.1, 3.5, 1.4, 0.2]
validated = InputValidator.validate_features(features)

# Raises InputValidationError for invalid inputs
InputValidator.validate_features([1, 2, 3])  # Wrong length
InputValidator.validate_features("invalid")   # Wrong type
```

### Run Tests

```bash
# All tests
poetry run pytest

# With coverage report
poetry run pytest --cov=src/ml_lambda --cov-report=term-missing

# Property-based tests only
poetry run pytest tests/property/
```

### Code Quality

```bash
poetry run lint
```

## Project Structure

```
ml-lambda-deployment/
├── src/ml_lambda/
│   ├── config.py           # Configuration dataclasses
│   ├── data/               # Data loading and preprocessing
│   │   └── processor.py    # DataProcessor with validation
│   ├── training/           # Model training and evaluation
│   │   ├── trainer.py      # ModelTrainer with cross-validation
│   │   └── evaluator.py    # ModelEvaluator for metrics
│   ├── model/              # Model serialization
│   │   └── serializer.py   # ModelSerializer with integrity checks
│   ├── inference/          # Lambda handler and validation
│   │   ├── validator.py    # InputValidator (NEW)
│   │   ├── predictor.py    # Predictor logic
│   │   └── handler.py      # Lambda handler (in progress)
│   ├── deploy/             # Packaging and AWS deployment
│   │   ├── packager.py     # ZIP packaging
│   │   └── deployer.py     # AWS deployment
│   └── utils/              # Logging and custom exceptions
│       ├── logging.py      # StructuredLogger
│       └── exceptions.py   # Custom exceptions
├── tests/
│   ├── unit/               # Unit tests
│   ├── property/           # Property-based tests (Hypothesis)
│   └── integration/        # Integration tests
├── scripts/                # CLI scripts
│   └── train.py            # Training script
├── artifacts/              # Trained models
└── docs/                   # Documentation
```

## API Reference

### POST /predict

Classify an Iris flower based on its features.

**Request**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Features (in order): sepal length, sepal width, petal length, petal width (cm)

**Response**

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [0.95, 0.03, 0.02],
  "latency_ms": 12.5
}
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and data flow
- [ML Concepts](docs/CONCEPTS.md) - Machine learning fundamentals

## Testing Strategy

The project uses a comprehensive testing approach:

- **Unit Tests**: Validate individual components
- **Property-Based Tests**: Use [Hypothesis](https://hypothesis.readthedocs.io/) to verify invariants across random inputs
- **Integration Tests**: Verify end-to-end workflows

## License

[MIT](LICENSE)

## Build by

- Jose Mosquera
