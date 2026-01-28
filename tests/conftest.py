"""Fixtures compartidos para pytest."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


@pytest.fixture
def iris_data():
    """Dataset Iris para tests."""
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def trained_model(iris_data):
    """Modelo RandomForest entrenado para tests."""
    X, y = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_features():
    """Features de ejemplo para predicción."""
    return [5.1, 3.5, 1.4, 0.2]


@pytest.fixture
def valid_api_event(sample_features):
    """Evento válido de API Gateway."""
    import json
    return {
        "body": json.dumps({"features": sample_features}),
        "httpMethod": "POST",
        "path": "/predict",
    }
