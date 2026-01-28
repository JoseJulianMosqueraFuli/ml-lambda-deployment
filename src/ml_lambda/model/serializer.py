"""Serialización y deserialización de modelos ML."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelMetadata:
    """Metadatos del modelo serializado."""

    version: str
    created_at: datetime
    accuracy: float
    n_features: int
    n_classes: int
    feature_names: list[str]
    class_names: list[str]
    training_config: dict[str, Any]


@dataclass
class SerializedModel:
    """Modelo serializado con metadatos."""

    model: Any
    metadata: ModelMetadata


class ModelSerializer:
    """Serializa y deserializa modelos ML."""

    def save(self, model, metadata: ModelMetadata, path: Path) -> str:
        """Guarda modelo con metadatos. Retorna hash SHA256."""
        # TODO: Implementar en tarea 9.2
        raise NotImplementedError

    def load(self, path: Path) -> SerializedModel:
        """Carga modelo con validación de integridad."""
        # TODO: Implementar en tarea 9.3
        raise NotImplementedError

    def validate_integrity(self, path: Path, expected_hash: str) -> bool:
        """Valida integridad del archivo."""
        # TODO: Implementar en tarea 9.3
        raise NotImplementedError
