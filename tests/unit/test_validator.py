"""Tests unitarios para el validador de entrada."""

import pytest
from ml_lambda.inference.validator import InputValidator, MAX_BODY_SIZE
from ml_lambda.utils.exceptions import InputValidationError


class TestInputValidator:
    """Tests para InputValidator."""

    def test_validate_features_valid(self):
        """Test de entrada válida."""
        features = [5.1, 3.5, 1.4, 0.2]
        result = InputValidator.validate_features(features)
        assert result == features
        assert all(isinstance(x, float) for x in result)

    def test_validate_features_converts_to_float(self):
        """Test de conversión a float."""
        features = [5, 3, 1, 0]  # ints
        result = InputValidator.validate_features(features)
        assert all(isinstance(x, float) for x in result)
        assert result == [5.0, 3.0, 1.0, 0.0]

    def test_validate_features_not_list(self):
        """Test de entrada que no es lista."""
        with pytest.raises(InputValidationError) as exc_info:
            InputValidator.validate_features("not a list")
        assert "debe ser una lista" in str(exc_info.value).lower()

    def test_validate_features_wrong_length(self):
        """Test de longitud incorrecta."""
        with pytest.raises(InputValidationError) as exc_info:
            InputValidator.validate_features([1.0, 2.0, 3.0])
        assert "4 features" in str(exc_info.value)

    def test_validate_features_invalid_type(self):
        """Test de tipo inválido en feature."""
        with pytest.raises(InputValidationError) as exc_info:
            InputValidator.validate_features([1.0, 2.0, "invalid", 4.0])
        assert "posición 2" in str(exc_info.value)

    def test_validate_features_out_of_range_warning(self, caplog):
        """Test de warning por valor fuera de rango."""
        features = [100.0, 3.5, 1.4, 0.2]  # sepal_length muy alto
        result = InputValidator.validate_features(features)
        assert result == features
        assert any("fuera de rango" in record.message for record in caplog.records)

    def test_validate_body_size_valid(self):
        """Test de body con tamaño válido."""
        body = '{"features": [1, 2, 3, 4]}'
        InputValidator.validate_body_size(body)  # No debe lanzar excepción

    def test_validate_body_size_too_large(self):
        """Test de body muy grande."""
        body = "x" * (MAX_BODY_SIZE + 1)
        with pytest.raises(InputValidationError) as exc_info:
            InputValidator.validate_body_size(body)
        assert "excede tamaño máximo" in str(exc_info.value).lower()

    def test_sanitize_input_string(self):
        """Test de sanitización de string."""
        dangerous = "<script>alert('xss')</script>"
        sanitized = InputValidator.sanitize_input(dangerous)
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "script" in sanitized  # El texto se mantiene

    def test_sanitize_input_non_string(self):
        """Test de sanitización de no-string."""
        value = 123
        result = InputValidator.sanitize_input(value)
        assert result == value

    def test_validate_features_with_none(self):
        """Test de feature con None."""
        with pytest.raises(InputValidationError):
            InputValidator.validate_features([1.0, 2.0, None, 4.0])

    def test_validate_features_empty_list(self):
        """Test de lista vacía."""
        with pytest.raises(InputValidationError) as exc_info:
            InputValidator.validate_features([])
        assert "4 features" in str(exc_info.value)
