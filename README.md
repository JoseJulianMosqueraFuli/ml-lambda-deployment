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
- **Serverless Inference**: Deploy as an AWS Lambda function behind API Gateway
- **Structured Logging**: JSON-formatted logs for observability

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
3. Evaluate metrics (accuracy, precision, recall, F1-score)
4. Save the model to `artifacts/`

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
│   ├── training/           # Model training and evaluation
│   ├── model/              # Model serialization
│   ├── inference/          # Lambda handler and validation
│   ├── deploy/             # Packaging and AWS deployment
│   └── utils/              # Logging and custom exceptions
├── tests/
│   ├── unit/               # Unit tests
│   ├── property/           # Property-based tests (Hypothesis)
│   └── integration/        # Integration tests
├── scripts/                # CLI scripts
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
