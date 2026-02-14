"""Serialización y deserialización de modelos ML."""

import hashlib
import joblib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_lambda.utils.exceptions import (
    ModelCorruptedError,
    ModelNotFoundError,
)


@dataclass
class ModelMetadata:
    """Metadatos del modelo serializado.

    Includes semantic version information, creation date,
    evaluation metrics and training configuration.

    Attributes:
        version: Versión semántica del modelo (ej: v1.0.0)
        created_at: Fecha y hora de creación del modelo
        accuracy: Accuracy del modelo en datos de test
        n_features: Número de features esperados por el modelo
        n_classes: Número de clases que predice el modelo
        feature_names: Nombres de las features de entrada
        class_names: Nombres de las clases de salida
        training_config: Configuración usada durante el entrenamiento
    """

    version: str
    created_at: datetime
    accuracy: float
    n_features: int
    n_classes: int
    feature_names: list[str]
    class_names: list[str]
    training_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convierte los metadatos a diccionario serializable."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Crea ModelMetadata desde un diccionario."""
        data = data.copy()
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class SerializedModel:
    """Modelo serializado con metadatos."""

    model: Any
    metadata: ModelMetadata


class ModelSerializer:
    """Serializa y deserializa modelos ML.

    Proporciona métodos para guardar y cargar modelos con sus metadatos,
    incluyendo validación de integridad mediante hash SHA256.
    """

    def save(self, model: Any, metadata: ModelMetadata, path: Path) -> str:
        """Guarda modelo con metadatos usando joblib.

        Args:
            model: Modelo entrenado a serializar
            metadata: Metadatos del modelo
            path: Ruta donde guardar el archivo

        Returns:
            Hash SHA256 del archivo guardado
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Crear objeto serializable con modelo y metadatos
        serialized_data = {
            "model": model,
            "metadata": metadata.to_dict(),
        }

        # Guardar con joblib
        joblib.dump(serialized_data, path)

        # Calcular y retornar hash SHA256
        return self._compute_hash(path)

    def load(self, path: Path) -> SerializedModel:
        """Carga modelo con validación de integridad.

        Args:
            path: Ruta al archivo del modelo

        Returns:
            SerializedModel with model and metadata

        Raises:
            ModelNotFoundError: If the file does not exist
            ModelCorruptedError: If the file is corrupted
        """
        path = Path(path)

        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {path}"
            )

        try:
            serialized_data = joblib.load(path)

            # Validar estructura del archivo
            if not isinstance(serialized_data, dict):
                raise ModelCorruptedError(
                    f"Invalid file format: expected dict, "
                    f"got {type(serialized_data).__name__}"
                )

            if "model" not in serialized_data:
                raise ModelCorruptedError(
                    "Corrupted file: missing 'model' field"
                )
            if "metadata" not in serialized_data:
                raise ModelCorruptedError(
                    "Corrupted file: missing 'metadata' field"
                )

            # Reconstruir metadatos
            metadata = ModelMetadata.from_dict(serialized_data["metadata"])

            return SerializedModel(
                model=serialized_data["model"],
                metadata=metadata,
            )

        except (ModelNotFoundError, ModelCorruptedError):
            raise
        except Exception as e:
            raise ModelCorruptedError(
                f"Error loading model: {e}"
            ) from e

    def validate_integrity(self, path: Path, expected_hash: str) -> bool:
        """Valida integridad del archivo comparando hash SHA256.

        Args:
            path: Ruta al archivo del modelo
            expected_hash: Hash SHA256 esperado

        Returns:
            True si el hash coincide, False en caso contrario
        """
        path = Path(path)

        if not path.exists():
            return False

        actual_hash = self._compute_hash(path)
        return actual_hash == expected_hash

    def _compute_hash(self, path: Path) -> str:
        """Calcula hash SHA256 del archivo.

        Args:
            path: Ruta al archivo

        Returns:
            Hash SHA256 como string hexadecimal
        """
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
