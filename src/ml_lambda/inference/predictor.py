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
        """Realiza predicción para un conjunto de features.
        
        Args:
            features: Lista de 4 features numéricas
            
        Returns:
            PredictionResult con predicción, nombre de clase y probabilidades
        """
        # Realizar predicción
        prediction = int(self._model.predict([features])[0])
        
        # Obtener probabilidades
        probabilities = self._model.predict_proba([features])[0].tolist()
        
        # Obtener nombre de clase
        class_name = self._class_names[prediction]
        
        return PredictionResult(
            prediction=prediction,
            class_name=class_name,
            probabilities=probabilities
        )
