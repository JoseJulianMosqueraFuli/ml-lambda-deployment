"""Property-based tests for model serialization round-trip.

Feature: ml-lambda-deployment, Property 4: Serialización Round-Trip Preserva Predicciones

**Validates: Requirements 5.4**
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from hypothesis import given, settings, strategies as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.ml_lambda.model.serializer import ModelMetadata, ModelSerializer


def _create_metadata() -> ModelMetadata:
    """Helper to create valid metadata for tests."""
    return ModelMetadata(
        version="v1.0.0",
        created_at=datetime.now(),
        accuracy=0.95,
        n_features=4,
        n_classes=3,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        class_names=["setosa", "versicolor", "virginica"],
        training_config={"n_estimators": 10, "random_state": 42},
    )


# Strategy for valid Iris-like features
features_strategy = st.lists(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=4,
)


class TestSerializationRoundTripProperties:
    """Property 4: Serialización Round-Trip Preserva Predicciones.

    Para cualquier modelo entrenado y cualquier conjunto de features válidos,
    si guardamos el modelo, lo cargamos, y hacemos predicciones,
    las predicciones deben ser idénticas a las del modelo original.

    **Validates: Requirements 5.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(features=features_strategy)
    def test_roundtrip_preserves_predictions(self, features):
        """
        Property 4: Para cualquier modelo y features válidos,
        save -> load debe preservar predicciones.

        **Validates: Requirements 5.4**
        """
        # Train a model
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Original prediction
        original_pred = model.predict([features])
        original_proba = model.predict_proba([features])

        # Round-trip: save then load
        serializer = ModelSerializer()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            serializer.save(model, _create_metadata(), model_path)
            loaded = serializer.load(model_path)

        # Prediction after round-trip
        loaded_pred = loaded.model.predict([features])
        loaded_proba = loaded.model.predict_proba([features])

        # Predictions must be identical
        np.testing.assert_array_equal(original_pred, loaded_pred)
        np.testing.assert_array_equal(original_proba, loaded_proba)

    @settings(max_examples=100, deadline=None)
    @given(features=features_strategy)
    def test_roundtrip_preserves_metadata(self, features):
        """
        Property 4: El round-trip debe preservar los metadatos del modelo.

        **Validates: Requirements 5.4**
        """
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        metadata = _create_metadata()

        serializer = ModelSerializer()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            serializer.save(model, metadata, model_path)
            loaded = serializer.load(model_path)

        assert loaded.metadata.version == metadata.version
        assert loaded.metadata.accuracy == metadata.accuracy
        assert loaded.metadata.n_features == metadata.n_features
        assert loaded.metadata.n_classes == metadata.n_classes
        assert loaded.metadata.feature_names == metadata.feature_names
        assert loaded.metadata.class_names == metadata.class_names
