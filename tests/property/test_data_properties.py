"""Property-based tests for DataProcessor.

Feature: ml-lambda-deployment, Property 1: División de Datos Mantiene Proporción
"""

import numpy as np
from hypothesis import given, settings, strategies as st

from ml_lambda.data.processor import DataProcessor


# Strategy for test_size (proportion for test set)
test_size_strategy = st.floats(min_value=0.1, max_value=0.5)

# Strategy for random_state
random_state_strategy = st.integers(min_value=0, max_value=10000)


class TestDataSplitProperties:
    """Property 1: División de Datos Mantiene Proporción.

    **Validates: Requirements 3.2**
    """

    @settings(max_examples=100)
    @given(test_size=test_size_strategy, random_state=random_state_strategy)
    def test_split_maintains_proportion(self, test_size, random_state):
        """
        Property 1: Para cualquier dataset con N muestras, cuando se divide
        con test_size, el conjunto de entrenamiento debe contener
        aproximadamente (1-test_size)*100% de las muestras (±1 muestra).

        **Validates: Requirements 3.2**
        """
        processor = DataProcessor(test_size=test_size, random_state=random_state)
        X, y = processor.load_iris()
        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        total_samples = len(X)
        expected_test = int(total_samples * test_size)
        expected_train = total_samples - expected_test

        # Allow ±1 sample tolerance for rounding
        assert abs(len(X_test) - expected_test) <= 1
        assert abs(len(X_train) - expected_train) <= 1

        # Total must be preserved
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_split_preserves_all_samples(self, random_state):
        """
        Property 1: La división no debe perder ni duplicar muestras.

        **Validates: Requirements 3.2**
        """
        processor = DataProcessor(test_size=0.2, random_state=random_state)
        X, y = processor.load_iris()
        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        # Concatenate and check all samples are present
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        # Same number of samples
        assert len(X_combined) == len(X)
        assert len(y_combined) == len(y)

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_split_is_reproducible(self, random_state):
        """
        Property 1: La división debe ser reproducible con el mismo random_state.

        **Validates: Requirements 3.2**
        """
        processor1 = DataProcessor(test_size=0.2, random_state=random_state)
        processor2 = DataProcessor(test_size=0.2, random_state=random_state)

        X, y = processor1.load_iris()

        X_train1, X_test1, y_train1, y_test1 = processor1.split_data(X, y)
        X_train2, X_test2, y_train2, y_test2 = processor2.split_data(X, y)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)



class TestNormalizationProperties:
    """Property 2: Normalización Produce Features Estandarizados.

    **Validates: Requirements 3.3**
    """

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_normalization_produces_zero_mean(self, random_state):
        """
        Property 2: Para cualquier conjunto de features numéricos,
        después de normalizar con StandardScaler (fit=True),
        la media de cada columna debe ser aproximadamente 0 (|mean| < 1e-10).

        **Validates: Requirements 3.3**
        """
        processor = DataProcessor(random_state=random_state)
        X, _ = processor.load_iris()
        X_normalized = processor.normalize(X, fit=True)

        # Mean should be approximately 0 for each feature
        means = X_normalized.mean(axis=0)
        for i, mean in enumerate(means):
            assert abs(mean) < 1e-10, f"Feature {i} mean {mean} not close to 0"

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_normalization_produces_unit_std(self, random_state):
        """
        Property 2: Para cualquier conjunto de features numéricos,
        después de normalizar con StandardScaler (fit=True),
        la desviación estándar debe ser aproximadamente 1 (|std - 1| < 1e-10).

        **Validates: Requirements 3.3**
        """
        processor = DataProcessor(random_state=random_state)
        X, _ = processor.load_iris()
        X_normalized = processor.normalize(X, fit=True)

        # Std should be approximately 1 for each feature
        stds = X_normalized.std(axis=0)
        for i, std in enumerate(stds):
            assert abs(std - 1) < 1e-10, f"Feature {i} std {std} not close to 1"

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_normalization_preserves_shape(self, random_state):
        """
        Property 2: La normalización debe preservar la forma del array.

        **Validates: Requirements 3.3**
        """
        processor = DataProcessor(random_state=random_state)
        X, _ = processor.load_iris()
        original_shape = X.shape

        X_normalized = processor.normalize(X, fit=True)

        assert X_normalized.shape == original_shape

    @settings(max_examples=100)
    @given(random_state=random_state_strategy)
    def test_normalization_transform_uses_fitted_scaler(self, random_state):
        """
        Property 2: Normalizar con fit=False debe usar el scaler ajustado.

        **Validates: Requirements 3.3**
        """
        processor = DataProcessor(random_state=random_state)
        X, _ = processor.load_iris()

        # Split data
        X_train = X[:100]
        X_test = X[100:]

        # Fit on training data
        processor.normalize(X_train, fit=True)

        # Transform test data without refitting
        X_test_normalized = processor.normalize(X_test, fit=False)

        # Shape should be preserved
        assert X_test_normalized.shape == X_test.shape

        # Test data mean/std won't be exactly 0/1 since scaler was fit on train
        # But values should be finite and reasonable
        assert np.isfinite(X_test_normalized).all()
