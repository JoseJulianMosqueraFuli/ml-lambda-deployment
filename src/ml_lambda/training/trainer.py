"""Entrenamiento de modelos de clasificación."""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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
        """Entrena el modelo con validación cruzada.

        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento

        Returns:
            TrainingResult con modelo y métricas de CV
        """
        start_time = time.perf_counter()

        # Crear modelo con configuración
        self._model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=self.config.random_state,
        )

        # Validación cruzada
        cv_scores = cross_val_score(
            self._model, X_train, y_train, cv=self.config.n_cv_folds
        )

        # Entrenar modelo final con todos los datos
        self._model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        return TrainingResult(
            model=self._model,
            training_time_seconds=training_time,
            cv_scores=cv_scores.tolist(),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
        )

    @property
    def model(self) -> RandomForestClassifier:
        """Retorna el modelo entrenado."""
        from ..utils.exceptions import ModelNotTrainedError

        if self._model is None:
            raise ModelNotTrainedError("Modelo no entrenado")
        return self._model
