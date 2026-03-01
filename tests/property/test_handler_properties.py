"""Property tests para LambdaHandler.

Feature: ml-lambda-deployment
Property 5: Solicitudes Válidas Producen Respuestas Completas
Property 11: Errores Internos No Exponen Detalles
"""

import json
import pytest
from hypothesis import given, strategies as st, settings
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
@settings(max_examples=100)
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
@settings(max_examples=100)
@given(
    error_message=st.text(min_size=10, max_size=100).filter(
        lambda x: "sensitive" in x.lower() or "password" in x.lower() or "secret" in x.lower()
    )
)
def test_property_internal_errors_dont_expose_details(handler_with_model, error_message, monkeypatch):
    """
    Property 11: Para cualquier error interno con información sensible,
    la respuesta debe:
    - Tener statusCode 500
    - NO contener el mensaje de error original
    - NO contener stack traces
    - Contener solo mensaje genérico "Internal server error"
    
    Validates: Requirements 12.2
    """
    # Arrange - Forzar error interno con mensaje sensible
    def mock_predict(*args, **kwargs):
        raise RuntimeError(error_message)
    
    monkeypatch.setattr(handler_with_model._predictor, "predict", mock_predict)
    
    event = {
        "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
    }
    context = Mock()
    context.aws_request_id = "test-request"
    
    # Act
    response = handler_with_model.handle(event, context)
    
    # Assert - Código de error interno
    assert response["statusCode"] == 500
    
    # Assert - Body parseado
    body = json.loads(response["body"])
    
    # Assert - Contiene campo errors
    assert "errors" in body
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) > 0
    
    # Assert - Solo mensaje genérico
    assert body["errors"] == ["Internal server error"]
    
    # Assert - No expone detalles internos
    response_str = json.dumps(response).lower()
    
    # No debe contener el mensaje de error original
    assert error_message.lower() not in response_str
    
    # No debe contener palabras sensibles del error
    sensitive_words = ["sensitive", "password", "secret", "traceback", "exception"]
    for word in sensitive_words:
        if word in error_message.lower():
            assert word not in response_str or word == "error"  # "error" es OK en "Internal server error"
    
    # No debe contener paths de archivos
    assert ".py" not in response_str
    assert "/src/" not in response_str
    assert "\\src\\" not in response_str


# Property adicional: Validación de entrada rechaza inputs inválidos
@settings(max_examples=50)
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
