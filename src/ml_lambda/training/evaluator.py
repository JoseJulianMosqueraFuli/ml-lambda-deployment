"""Evaluación de modelos de clasificación."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: str


class ModelEvaluator:
    """Evalúa modelos de clasificación."""

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Evalúa el modelo en datos de test."""
        # TODO: Implementar en tarea 7.4
        raise NotImplementedError

    def check_accuracy_threshold(
        self, metrics: EvaluationMetrics, threshold: float = 0.9
    ) -> bool:
        """Verifica si accuracy supera umbral."""
        # TODO: Implementar en tarea 7.5
        raise NotImplementedError
