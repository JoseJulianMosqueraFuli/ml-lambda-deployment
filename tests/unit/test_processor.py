"""Tests unitarios para DataProcessor."""

import numpy as np
import pytest

from ml_lambda.data.processor import DataProcessor, DatasetStats
from ml_lambda.utils.exceptions import DataValidationError


class TestDataProcessorLoadIris:
    """Tests para carga del dataset Iris."""

    def test_load_iris_returns_tuple(self):
        """Verifica que load_iris retorna una tupla de arrays."""
        processor = DataProcessor()
        X, y = processor.load_iris()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_load_iris_correct_shape(self):
        """Verifica que el dataset tiene la forma correcta (150 samples, 4 features)."""
        processor = DataProcessor()
        X, y = processor.load_iris()

        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_load_iris_no_nulls(self):
        """Verifica que el dataset no contiene valores nulos."""
        processor = DataProcessor()
        X, y = processor.load_iris()

        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestDataProcessorSplitData:
    """Tests para división de datos."""

    def test_split_data_returns_four_arrays(self):
        """Verifica que split_data retorna 4 arrays."""
        processor = DataProcessor()
        X, y = processor.load_iris()
        result = processor.split_data(X, y)

        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_split_data_correct_proportions(self):
        """Verifica que la división es aproximadamente 80/20."""
        processor = DataProcessor(test_size=0.2)
        X, y = processor.load_iris()
        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        total = len(X)
        train_ratio = len(X_train) / total
        test_ratio = len(X_test) / total

        assert 0.79 <= train_ratio <= 0.81
        assert 0.19 <= test_ratio <= 0.21

    def test_split_data_reproducible(self):
        """Verifica que la división es reproducible con el mismo random_state."""
        processor1 = DataProcessor(random_state=42)
        processor2 = DataProcessor(random_state=42)

        X, y = processor1.load_iris()
        X_train1, X_test1, _, _ = processor1.split_data(X, y)
        X_train2, X_test2, _, _ = processor2.split_data(X, y)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)


class TestDataProcessorNormalize:
    """Tests para normalización."""

    def test_normalize_with_fit(self):
        """Verifica que normalize con fit=True ajusta el scaler."""
        processor = DataProcessor()
        X, _ = processor.load_iris()
        X_normalized = processor.normalize(X, fit=True)

        assert X_normalized.shape == X.shape
        assert processor._scaler is not None

    def test_normalize_produces_standard_distribution(self):
        """Verifica que la normalización produce media ~0 y std ~1."""
        processor = DataProcessor()
        X, _ = processor.load_iris()
        X_normalized = processor.normalize(X, fit=True)

        # Media cercana a 0
        means = X_normalized.mean(axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros(4), decimal=10)

        # Std cercana a 1
        stds = X_normalized.std(axis=0)
        np.testing.assert_array_almost_equal(stds, np.ones(4), decimal=10)

    def test_normalize_without_fit_uses_existing_scaler(self):
        """Verifica que normalize sin fit usa el scaler existente."""
        processor = DataProcessor()
        X, _ = processor.load_iris()

        # Primero ajustar con datos de entrenamiento
        X_train = X[:100]
        processor.normalize(X_train, fit=True)

        # Luego transformar datos de test sin reajustar
        X_test = X[100:]
        X_test_normalized = processor.normalize(X_test, fit=False)

        assert X_test_normalized.shape == X_test.shape


class TestDataProcessorComputeStats:
    """Tests para cálculo de estadísticas."""

    def test_compute_stats_returns_dataclass(self):
        """Verifica que compute_stats retorna DatasetStats."""
        processor = DataProcessor()
        X, y = processor.load_iris()
        stats = processor.compute_stats(X, y)

        assert isinstance(stats, DatasetStats)

    def test_compute_stats_correct_values(self):
        """Verifica que las estadísticas son correctas para Iris."""
        processor = DataProcessor()
        X, y = processor.load_iris()
        stats = processor.compute_stats(X, y)

        assert stats.n_samples == 150
        assert stats.n_features == 4
        assert stats.n_classes == 3
        assert sum(stats.class_distribution.values()) == 150
        assert len(stats.feature_ranges) == 4

    def test_compute_stats_class_distribution(self):
        """Verifica la distribución de clases (50 por clase en Iris)."""
        processor = DataProcessor()
        X, y = processor.load_iris()
        stats = processor.compute_stats(X, y)

        # Iris tiene 50 muestras por clase
        for class_id, count in stats.class_distribution.items():
            assert count == 50
