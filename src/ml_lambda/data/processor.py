"""Procesamiento y preparación de datos."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
        """Carga el dataset Iris."""
        # TODO: Implementar en tarea 5.1
        raise NotImplementedError

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide datos en train/test."""
        # TODO: Implementar en tarea 5.2
        raise NotImplementedError

    def normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza features usando StandardScaler."""
        # TODO: Implementar en tarea 5.3
        raise NotImplementedError

    def compute_stats(self, X: np.ndarray, y: np.ndarray) -> DatasetStats:
        """Calcula estadísticas del dataset."""
        # TODO: Implementar en tarea 5.4
        raise NotImplementedError
