"""Lógica de predicción."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    """Resultado de predicción."""

    prediction: int
    class_name: str
    probabilities: list[float]


class Predictor:
    """Realiza predicciones con el modelo cargado."""

    def __init__(self, model: Any, class_names: list[str]):
        self._model = model
        self._class_names = class_names

    def predict(self, features: list[float]) -> PredictionResult:
        """Realiza predicción para un conjunto de features."""
        # TODO: Implementar en tarea 14.1
        raise NotImplementedError
