"""Validación de entradas de la API."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Resultado de validación."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    sanitized_input: Any


class InputValidator:
    """Valida entradas de la API de inferencia."""

    EXPECTED_FEATURES = 4
    MAX_BODY_SIZE = 1024  # 1KB

    # Rangos típicos del dataset Iris
    FEATURE_RANGES = [
        (4.0, 8.0),  # sepal length
        (2.0, 4.5),  # sepal width
        (1.0, 7.0),  # petal length
        (0.1, 2.5),  # petal width
    ]

    def validate(self, body: dict[str, Any]) -> ValidationResult:
        """Valida el body de la solicitud."""
        # TODO: Implementar en tarea 12.1
        raise NotImplementedError

    def _validate_features(self, features: Any) -> tuple[bool, list[str], list[str]]:
        """Valida el campo features."""
        # TODO: Implementar en tarea 12.2
        raise NotImplementedError

    def _check_feature_ranges(self, features: list[float]) -> list[str]:
        """Genera warnings si features están fuera de rango típico."""
        # TODO: Implementar en tarea 12.4
        raise NotImplementedError
