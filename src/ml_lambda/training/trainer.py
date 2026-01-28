"""Entrenamiento de modelos de clasificación."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    random_state: int = 42
    n_cv_folds: int = 5


@dataclass
class TrainingResult:
    """Resultado del entrenamiento."""

    model: RandomForestClassifier
    training_time_seconds: float
    cv_scores: list[float]
    cv_mean: float
    cv_std: float


class ModelTrainer:
    """Entrena modelos de clasificación."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._model: Optional[RandomForestClassifier] = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> TrainingResult:
        """Entrena el modelo con validación cruzada."""
        # TODO: Implementar en tarea 7.2
        raise NotImplementedError

    @property
    def model(self) -> RandomForestClassifier:
        """Retorna el modelo entrenado."""
        from ..utils.exceptions import ModelNotTrainedError

        if self._model is None:
            raise ModelNotTrainedError("Modelo no entrenado")
        return self._model
