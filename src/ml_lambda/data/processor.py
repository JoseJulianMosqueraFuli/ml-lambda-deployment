"""Procesamiento y preparación de datos."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_lambda.utils.exceptions import DataValidationError


@dataclass
class DatasetStats:
    """Estadísticas del dataset."""

    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: dict[int, int]
    feature_ranges: list[Tuple[float, float]]


class DataProcessor:
    """Procesa y prepara datos para entrenamiento."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self._scaler = None

    def load_iris(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carga el dataset Iris.

        Returns:
            Tuple de (features, labels)

        Raises:
            DataValidationError: Si los datos contienen valores nulos
        """
        iris = sklearn_load_iris()
        X, y = iris.data, iris.target

        # Validar que no hay valores nulos
        if np.isnan(X).any():
            raise DataValidationError("Dataset contiene valores nulos en features")
        if np.isnan(y).any():
            raise DataValidationError("Dataset contiene valores nulos en labels")

        return X, y

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide datos en train/test.

        Args:
            X: Features del dataset
            y: Labels del dataset

        Returns:
            Tuple de (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza features usando StandardScaler.

        Args:
            X: Features a normalizar
            fit: Si True, ajusta el scaler; si False, usa scaler existente

        Returns:
            Features normalizados
        """
        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return self._scaler.transform(X)

    def compute_stats(self, X: np.ndarray, y: np.ndarray) -> DatasetStats:
        """Calcula estadísticas del dataset.

        Args:
            X: Features del dataset
            y: Labels del dataset

        Returns:
            DatasetStats con n_samples, distribución, rangos
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # Distribución de clases
        class_distribution = {int(c): int(np.sum(y == c)) for c in unique_classes}

        # Rangos de features
        feature_ranges = [
            (float(X[:, i].min()), float(X[:, i].max())) for i in range(n_features)
        ]

        return DatasetStats(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=class_distribution,
            feature_ranges=feature_ranges,
        )
