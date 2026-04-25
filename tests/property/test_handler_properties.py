"""Property tests para LambdaHandler.

Feature: ml-lambda-deployment
Property 5: Solicitudes Válidas Producen Respuestas Completas
Property 11: Errores Internos No Exponen Detalles
"""

import json
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock

from ml_lambda.inference.handler import LambdaHandler


@pytest.fixture
def handler_with_model(trained_model, tmp_path, monkeypatch):
    """Handler con modelo cargado."""
    from ml_lambda.model.serializer import ModelSerializer, ModelMetadata
    from datetime import datetime
    
    # Crear y guardar modelo
    serializer = ModelSerializer()
    metadata = ModelMetadata(
        version="v1.0.0",
        created_at=datetime.now(),
        accuracy=0.95,
        n_features=4,
        n_classes=3,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        class_names=["setosa", "versicolor", "virginica"],
        training_config={}
    )
    
    model_path = tmp_path / "model.joblib"
    serializer.save(trained_model, metadata, model_path)
    
    # Monkeypatch config para usar directorio temporal
    from ml_lambda import config
    monkeypatch.setattr(config.config, "artifacts_dir", tmp_path)
    monkeypatch.setattr(config.config, "model_filename", "model.joblib")
    
    return LambdaHandler()


# Property 5: Solicitudes Válidas Producen Respuestas Completas
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    features=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4
    )
)
def test_property_valid_requests_produce_complete_responses(handler_with_model, features):
    """
    Property 5: Para cualquier solicitud con features válidos (lista de 4 números),
    la respuesta debe:
    - Tener statusCode 200
    - Contener campo "prediction" (entero 0, 1, o 2)
    - Contener campo "probabilities" (lista de 3 floats que suman ~1.0)
    - Contener campo "latency_ms" (número positivo)
    
    Validates: Requirements 6.1, 6.2
    """
    # Arrange
    event = {
        "body": json.dumps({"features": features})
    }
    context = Mock()
    context.aws_request_id = "test-request"
    
    # Act
    response = handler_with_model.handle(event, context)
    
    # Assert - Estructura de respuesta
    assert "statusCode" in response
    assert "headers" in response
    assert "body" in response
    
    # Assert - Código de éxito
    assert response["statusCode"] == 200
    
    # Assert - Headers
    assert "Content-Type" in response["headers"]
    assert response["headers"]["Content-Type"] == "application/json"
    
    # Assert - Body parseado
    body = json.loads(response["body"])
    
    # Assert - Campos requeridos presentes
    assert "prediction" in body
    assert "class_name" in body
    assert "probabilities" in body
    assert "latency_ms" in body
    
    # Assert - Prediction es entero válido
    assert isinstance(body["prediction"], int)
    assert 0 <= body["prediction"] <= 2
    
    # Assert - Class name es válido
    assert body["class_name"] in ["setosa", "versicolor", "virginica"]
    
    # Assert - Probabilities es lista de 3 floats
    assert isinstance(body["probabilities"], list)
    assert len(body["probabilities"]) == 3
    assert all(isinstance(p, float) for p in body["probabilities"])
    
    # Assert - Probabilities suman ~1.0
    prob_sum = sum(body["probabilities"])
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected ~1.0"
    
    # Assert - Todas las probabilidades están en [0, 1]
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])
    
    # Assert - Latency es positivo
    assert isinstance(body["latency_ms"], (int, float))
    assert body["latency_ms"] > 0


# Property 11: Errores Internos No Exponen Detalles
#
# Strategy: generate arbitrary error messages that could contain sensitive info
# such as file paths, credentials, stack traces, or internal details.
# The handler must never leak any of these in the HTTP response.

_internal_error_messages = st.one_of(
    # Arbitrary text that may contain anything
    st.text(min_size=1, max_size=200),
    # Messages resembling file paths
    st.from_regex(r"/[a-z_]+/[a-z_]+\.py line \d+", fullmatch=True),
    # Messages with credential-like content
    st.tuples(
        st.sampled_from(["password", "secret", "token", "api_key", "credential"]),
        st.text(min_size=1, max_size=30),
    ).map(lambda t: f"{t[0]}={t[1]}"),
    # Messages resembling stack traces
    st.just('Traceback (most recent call last):\n  File "handler.py", line 42\nKeyError: "model"'),
    # Messages with internal class/module names
    st.sampled_from([
        "ModelSerializer.load() failed: corrupt header at byte 0x3F",
        "NoneType has no attribute 'predict' in /src/ml_lambda/inference/predictor.py:28",
        "Connection to db://internal-host:5432 refused, password=hunter2",
        "sklearn.ensemble._forest.RandomForestClassifier raised ValueError",
        "AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE",
    ]),
)

_internal_exception_types = st.sampled_from([
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    MemoryError,
])


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    error_message=_internal_error_messages,
    exception_type=_internal_exception_types,
    features=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ),
)
def test_property_internal_errors_dont_expose_details(
    handler_with_model, error_message, exception_type, features, monkeypatch
):
    """
    Property 11: Para cualquier error interno (excepción no manejada),
    la respuesta HTTP debe:
    - Tener statusCode 500
    - NO contener stack trace en el body
    - NO contener nombres de archivos internos
    - NO contener el mensaje de error original (cuando es no trivial)
    - Contener solo mensaje genérico "Internal server error"

    Validates: Requirements 12.2
    """
    # Arrange – ensure model is loaded so _predictor exists, then force error
    if handler_with_model._predictor is None:
        handler_with_model._load_model_once()

    def mock_predict(*args, **kwargs):
        raise exception_type(error_message)

    monkeypatch.setattr(handler_with_model._predictor, "predict", mock_predict)

    event = {"body": json.dumps({"features": features})}
    context = Mock()
    context.aws_request_id = "test-request"

    # Act
    response = handler_with_model.handle(event, context)

    # Assert – status code is 500
    assert response["statusCode"] == 500

    # Assert – body is valid JSON
    body = json.loads(response["body"])

    # Assert – only the generic error message is present
    assert "errors" in body
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) == 1
    assert body["errors"] == ["Internal server error"]

    # Assert – the raw response string does not leak internal details
    response_body_str = response["body"]

    # Must not contain the original error message (skip trivially short ones
    # that could collide with the generic message words)
    if len(error_message) > 5 and error_message.lower() not in "internal server error":
        assert error_message not in response_body_str

    # Must not contain stack-trace indicators
    assert "Traceback" not in response_body_str
    assert "File \"" not in response_body_str

    # Must not contain Python file references
    assert ".py" not in response_body_str

    # Must not contain source path fragments
    assert "/src/" not in response_body_str
    assert "\\src\\" not in response_body_str
    assert "ml_lambda" not in response_body_str

    # Must not contain credential-like patterns
    for keyword in ("password", "secret", "token", "api_key", "AWS_SECRET"):
        assert keyword.lower() not in response_body_str.lower()


# Property adicional: Validación de entrada rechaza inputs inválidos
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    invalid_input=st.one_of(
        st.text(),  # String en lugar de lista
        st.integers(),  # Entero en lugar de lista
        st.lists(st.floats(), min_size=0, max_size=3),  # Lista muy corta
        st.lists(st.floats(), min_size=5, max_size=10),  # Lista muy larga
        st.lists(st.text(), min_size=4, max_size=4),  # Lista de strings
    )
)
def test_property_invalid_inputs_rejected(handler_with_model, invalid_input):
    """
    Property adicional: Para cualquier entrada inválida,
    el sistema debe retornar código 400 o 500.
    
    Validates: Requirements 6.4, 6.7, 7.1, 7.3
    """
    # Arrange
    event = {
        "body": json.dumps({"features": invalid_input})
    }
    context = Mock()
    context.aws_request_id = "test-request"
    
    # Act
    response = handler_with_model.handle(event, context)
    
    # Assert - Código de error (400 o 500)
    assert response["statusCode"] in [400, 500]
    
    # Assert - Contiene errores
    body = json.loads(response["body"])
    assert "errors" in body
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) > 0
