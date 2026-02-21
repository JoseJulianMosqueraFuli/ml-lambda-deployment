"""Property tests para validación de entrada."""

from hypothesis import given, strategies as st
import pytest
from ml_lambda.inference.validator import InputValidator, MAX_BODY_SIZE
from ml_lambda.utils.exceptions import InputValidationError


class TestValidatorProperties:
    """Property tests para InputValidator."""

    @given(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        )
    )
    def test_property_valid_features_accepted(self, features):
        """Property 6: Validación acepta listas válidas de 4 floats."""
        result = InputValidator.validate_features(features)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    @given(
        st.one_of(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=3),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=5, max_size=10),
        )
    )
    def test_property_wrong_length_rejected(self, features):
        """Property 6: Validación rechaza listas con longitud != 4."""
        with pytest.raises(InputValidationError):
            InputValidator.validate_features(features)

    @given(st.one_of(st.integers(), st.text(), st.dictionaries(st.text(), st.integers())))
    def test_property_non_list_rejected(self, value):
        """Property 6: Validación rechaza entradas que no son listas."""
        with pytest.raises(InputValidationError):
            InputValidator.validate_features(value)

    @given(
        st.lists(
            st.one_of(st.floats(min_value=-100, max_value=100), st.text()),
            min_size=4,
            max_size=4,
        )
    )
    def test_property_invalid_types_rejected(self, features):
        """Property 6: Validación rechaza listas con tipos inválidos."""
        # Si todos son números, debe pasar
        if all(isinstance(x, (int, float)) for x in features):
            result = InputValidator.validate_features(features)
            assert len(result) == 4
        else:
            # Si hay strings, debe fallar
            with pytest.raises(InputValidationError):
                InputValidator.validate_features(features)

    @given(
        st.lists(
            st.floats(min_value=100, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        )
    )
    def test_property_out_of_range_generates_warnings(self, features, caplog):
        """Property 7: Valores fuera de rango generan warnings."""
        result = InputValidator.validate_features(features)
        assert len(result) == 4
        # Debe haber al menos un warning porque todos los valores son > 100
        assert any("fuera de rango" in record.message for record in caplog.records)

    @given(st.text(min_size=0, max_size=MAX_BODY_SIZE))
    def test_property_body_within_limit_accepted(self, body):
        """Property 8: Bodies dentro del límite son aceptados."""
        InputValidator.validate_body_size(body)  # No debe lanzar excepción

    @given(st.text(min_size=MAX_BODY_SIZE + 1, max_size=MAX_BODY_SIZE * 2))
    def test_property_body_exceeding_limit_rejected(self, body):
        """Property 8: Bodies que exceden el límite son rechazados."""
        with pytest.raises(InputValidationError):
            InputValidator.validate_body_size(body)

    @given(st.text())
    def test_property_sanitize_removes_dangerous_chars(self, text):
        """Property: Sanitización remueve caracteres peligrosos."""
        sanitized = InputValidator.sanitize_input(text)
        dangerous_chars = ["<", ">", "&", '"', "'", ";", "(", ")", "{", "}"]
        for char in dangerous_chars:
            assert char not in sanitized
