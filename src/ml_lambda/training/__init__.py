"""Módulo de entrenamiento de modelos."""

from .trainer import ModelTrainer, TrainingConfig, TrainingResult
from .evaluator import ModelEvaluator, EvaluationMetrics

__all__ = [
    "ModelTrainer",
    "TrainingConfig", 
    "TrainingResult",
    "ModelEvaluator",
    "EvaluationMetrics",
]
