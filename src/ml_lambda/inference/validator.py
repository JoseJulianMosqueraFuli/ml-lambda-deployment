"""Validador de entrada para la API de inferencia."""

from typing import Any, List
from ..utils.exceptions import InputValidationError
from ..utils.logging import StructuredLogger

logger = StructuredLogger()

# Rangos típicos del dataset Iris (en cm)
IRIS_RANGES = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.0, 3.0),
}

MAX_BODY_SIZE = 1024  # 1KB


class InputValidator:
    """Valida entrada de la API de inferencia."""

    @staticmethod
    def validate_features(features: Any) -> List[float]:
        """Valida y convierte features a lista de floats.

        Args:
            features: Entrada a validar

        Returns:
            Lista de 4 floats validados

        Raises:
            InputValidationError: Si la entrada es inválida
        """
        # Validar que sea lista
        if not isinstance(features, list):
            raise InputValidationError(
                f"Features debe ser una lista, recibido: {type(features).__name__}"
            )

        # Validar longitud
        if len(features) != 4:
            raise InputValidationError(
                f"Se esperan 4 features, recibidos: {len(features)}"
            )

        # Validar y convertir tipos
        validated = []
        for i, value in enumerate(features):
            try:
                validated.append(float(value))
            except (ValueError, TypeError) as e:
                raise InputValidationError(
                    f"Feature en posición {i} no es un número válido: {value}"
                ) from e

        # Validar rangos (warnings, no errores)
        InputValidator._check_ranges(validated)

        return validated

    @staticmethod
    def _check_ranges(features: List[float]) -> None:
        """Verifica si features están en rangos típicos de Iris.

        Args:
            features: Lista de 4 features validados
        """
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        for i, (name, value) in enumerate(zip(feature_names, features)):
            min_val, max_val = IRIS_RANGES[name]
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"Feature '{name}' fuera de rango típico",
                    extra={
                        "feature_index": i,
                        "feature_name": name,
                        "value": value,
                        "expected_range": [min_val, max_val],
                    },
                )

    @staticmethod
    def validate_body_size(body: str) -> None:
        """Valida que el body no exceda el tamaño máximo.

        Args:
            body: Body de la solicitud como string

        Raises:
            InputValidationError: Si el body excede 1KB
        """
        size = len(body.encode("utf-8"))
        if size > MAX_BODY_SIZE:
            raise InputValidationError(
                f"Body excede tamaño máximo de {MAX_BODY_SIZE} bytes: {size} bytes"
            )

    @staticmethod
    def sanitize_input(value: Any) -> Any:
        """Sanitiza entrada para prevenir inyección.

        Args:
            value: Valor a sanitizar

        Returns:
            Valor sanitizado
        """
        if isinstance(value, str):
            # Remover caracteres potencialmente peligrosos
            dangerous_chars = ["<", ">", "&", '"', "'", ";", "(", ")", "{", "}"]
            sanitized = value
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, "")
            return sanitized
        return value
