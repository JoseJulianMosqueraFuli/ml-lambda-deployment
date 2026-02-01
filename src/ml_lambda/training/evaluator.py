"""Evaluación de modelos de clasificación."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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

    def evaluate(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evalúa el modelo en datos de test.

        Args:
            model: Modelo entrenado con método predict
            X_test: Features de test
            y_test: Labels de test

        Returns:
            EvaluationMetrics con todas las métricas
        """
        y_pred = model.predict(X_test)

        return EvaluationMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, average="weighted")),
            recall=float(recall_score(y_test, y_pred, average="weighted")),
            f1_score=float(f1_score(y_test, y_pred, average="weighted")),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(y_test, y_pred),
        )

    def check_accuracy_threshold(
        self, metrics: EvaluationMetrics, threshold: float = 0.9
    ) -> bool:
        """Verifica si accuracy supera umbral.

        Args:
            metrics: Métricas de evaluación
            threshold: Umbral mínimo de accuracy

        Returns:
            True si accuracy >= threshold

        Warns:
            Si accuracy < threshold
        """
        if metrics.accuracy < threshold:
            warnings.warn(
                f"Accuracy {metrics.accuracy:.4f} está por debajo del umbral {threshold}",
                UserWarning,
            )
            return False
        return True
