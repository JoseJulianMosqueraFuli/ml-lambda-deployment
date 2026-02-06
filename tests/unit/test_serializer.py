"""Tests unitarios para ModelSerializer.

**Validates: Requirements 5.1, 5.2, 5.3, 5.5**
"""

from datetime import datetime

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.ml_lambda.model.serializer import (
    ModelMetadata,
    ModelSerializer,
    SerializedModel,
)
from src.ml_lambda.utils.exceptions import (
    ModelCorruptedError,
    ModelNotFoundError,
)


class TestModelSerializer:
    """Tests para ModelSerializer."""

    @pytest.fixture
    def trained_model(self):
        """Fixture con modelo entrenado."""
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def sample_metadata(self):
        """Fixture con metadatos de ejemplo."""
        return ModelMetadata(
            version="v1.0.0",
            created_at=datetime.now(),
            accuracy=0.95,
            n_features=4,
            n_classes=3,
            feature_names=["f1", "f2", "f3", "f4"],
            class_names=["setosa", "versicolor", "virginica"],
            training_config={"n_estimators": 10, "random_state": 42},
        )

    def test_save_creates_file(self, trained_model, sample_metadata, tmp_path):
        """Verifica que save() crea el archivo."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"

        serializer.save(trained_model, sample_metadata, model_path)

        assert model_path.exists()

    def test_save_returns_hash(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica que save() retorna hash SHA256."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"

        hash_value = serializer.save(
            trained_model, sample_metadata, model_path
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length

    def test_load_returns_serialized_model(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica que load() retorna SerializedModel."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"
        serializer.save(trained_model, sample_metadata, model_path)

        result = serializer.load(model_path)

        assert isinstance(result, SerializedModel)

    def test_load_preserves_model(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica que load() preserva el modelo."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"
        serializer.save(trained_model, sample_metadata, model_path)

        result = serializer.load(model_path)

        # El modelo cargado debe poder hacer predicciones
        X, _ = load_iris(return_X_y=True)
        predictions = result.model.predict(X[:5])
        assert len(predictions) == 5

    def test_load_preserves_metadata(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica que load() preserva los metadatos."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"
        serializer.save(trained_model, sample_metadata, model_path)

        result = serializer.load(model_path)

        assert result.metadata.version == sample_metadata.version
        assert result.metadata.accuracy == sample_metadata.accuracy
        assert result.metadata.n_features == sample_metadata.n_features
        assert result.metadata.n_classes == sample_metadata.n_classes

    def test_load_raises_on_missing_file(self, tmp_path):
        """Verifica que load() lanza error si archivo no existe."""
        serializer = ModelSerializer()
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(ModelNotFoundError):
            serializer.load(model_path)

    def test_load_raises_on_corrupted_file(self, tmp_path):
        """Verifica que load() lanza error si archivo está corrupto."""
        serializer = ModelSerializer()
        model_path = tmp_path / "corrupted.joblib"

        # Crear archivo corrupto
        model_path.write_text("not a valid joblib file")

        with pytest.raises(ModelCorruptedError):
            serializer.load(model_path)

    def test_validate_integrity_returns_true_for_valid_hash(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica que validate_integrity() retorna True para hash válido."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"
        expected_hash = serializer.save(
            trained_model, sample_metadata, model_path
        )

        result = serializer.validate_integrity(model_path, expected_hash)

        assert result is True

    def test_validate_integrity_returns_false_for_invalid_hash(
        self, trained_model, sample_metadata, tmp_path
    ):
        """Verifica validate_integrity() retorna False para hash inválido."""
        serializer = ModelSerializer()
        model_path = tmp_path / "model.joblib"
        serializer.save(trained_model, sample_metadata, model_path)

        result = serializer.validate_integrity(model_path, "invalid_hash")

        assert result is False

    def test_validate_integrity_returns_false_for_missing_file(
        self, tmp_path
    ):
        """Verifica validate_integrity() retorna False si archivo no existe."""
        serializer = ModelSerializer()
        model_path = tmp_path / "nonexistent.joblib"

        result = serializer.validate_integrity(model_path, "any_hash")

        assert result is False


class TestModelMetadata:
    """Tests para ModelMetadata."""

    def test_to_dict_serializes_datetime(self):
        """Verifica que to_dict() serializa datetime correctamente."""
        now = datetime.now()
        metadata = ModelMetadata(
            version="v1.0.0",
            created_at=now,
            accuracy=0.95,
            n_features=4,
            n_classes=3,
            feature_names=["f1", "f2", "f3", "f4"],
            class_names=["c1", "c2", "c3"],
            training_config={},
        )

        result = metadata.to_dict()

        assert result["created_at"] == now.isoformat()

    def test_from_dict_deserializes_datetime(self):
        """Verifica from_dict() deserializa datetime correctamente."""
        now = datetime.now()
        data = {
            "version": "v1.0.0",
            "created_at": now.isoformat(),
            "accuracy": 0.95,
            "n_features": 4,
            "n_classes": 3,
            "feature_names": ["f1", "f2", "f3", "f4"],
            "class_names": ["c1", "c2", "c3"],
            "training_config": {},
        }

        result = ModelMetadata.from_dict(data)

        assert isinstance(result.created_at, datetime)
