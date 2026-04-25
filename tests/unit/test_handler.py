"""Tests unitarios para LambdaHandler."""

import json
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from ml_lambda.inference.handler import LambdaHandler
from ml_lambda.utils.exceptions import InputValidationError


@pytest.fixture
def mock_context():
    """Mock del contexto de Lambda."""
    context = Mock()
    context.aws_request_id = "test-request-123"
    return context


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


class TestLambdaHandler:
    """Tests para LambdaHandler."""
    
    def test_handle_valid_request(self, handler_with_model, mock_context):
        """Test de solicitud válida."""
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 200
        assert "Content-Type" in response["headers"]
        assert response["headers"]["Access-Control-Allow-Origin"] == "*"
        
        body = json.loads(response["body"])
        assert "prediction" in body
        assert "class_name" in body
        assert "probabilities" in body
        assert "latency_ms" in body
        
        assert isinstance(body["prediction"], int)
        assert 0 <= body["prediction"] <= 2
        assert body["class_name"] in ["setosa", "versicolor", "virginica"]
        assert len(body["probabilities"]) == 3
        assert abs(sum(body["probabilities"]) - 1.0) < 0.01
        assert body["latency_ms"] > 0
    
    def test_handle_direct_dict_event(self, handler_with_model, mock_context):
        """Test con evento como dict directo (sin body string)."""
        event = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "prediction" in body
    
    def test_handle_missing_features(self, handler_with_model, mock_context):
        """Test de solicitud sin campo features."""
        event = {
            "body": json.dumps({"data": [1, 2, 3, 4]})
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "errors" in body
        assert "features" in body["errors"][0].lower()
    
    def test_handle_invalid_features_type(self, handler_with_model, mock_context):
        """Test de features con tipo inválido."""
        event = {
            "body": json.dumps({"features": "not a list"})
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "errors" in body
    
    def test_handle_invalid_features_length(self, handler_with_model, mock_context):
        """Test de features con longitud incorrecta."""
        event = {
            "body": json.dumps({"features": [1, 2, 3]})  # Solo 3 features
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "errors" in body
    
    def test_handle_invalid_json(self, handler_with_model, mock_context):
        """Test de JSON malformado."""
        event = {
            "body": "not valid json {"
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "errors" in body
        assert body["errors"][0] == "Internal server error"
    
    def test_handle_body_too_large(self, handler_with_model, mock_context):
        """Test de body que excede tamaño máximo."""
        large_body = json.dumps({"features": [1.0, 2.0, 3.0, 4.0], "extra": "x" * 2000})
        event = {
            "body": large_body
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        # InputValidationError retorna 400, no 500
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "errors" in body
    
    def test_handle_without_context(self, handler_with_model):
        """Test sin contexto de Lambda (ejecución local)."""
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        
        response = handler_with_model.handle(event, None)
        
        assert response["statusCode"] == 200
    
    def test_model_loaded_once(self, handler_with_model, mock_context):
        """Test que el modelo se carga solo una vez."""
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        
        # Primera invocación
        response1 = handler_with_model.handle(event, mock_context)
        assert response1["statusCode"] == 200
        
        # Segunda invocación (warm start)
        response2 = handler_with_model.handle(event, mock_context)
        assert response2["statusCode"] == 200
        
        # El modelo debe estar cargado
        assert handler_with_model._model is not None
        assert handler_with_model._predictor is not None
    
    def test_cors_headers_present(self, handler_with_model, mock_context):
        """Test que los headers CORS están presentes."""
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        headers = response["headers"]
        assert "Access-Control-Allow-Origin" in headers
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "Access-Control-Allow-Methods" in headers
        assert "Access-Control-Allow-Headers" in headers
    
    def test_error_response_no_internal_details(self, handler_with_model, mock_context, monkeypatch):
        """Test que errores internos no exponen detalles."""
        # Primero cargar el modelo
        event_init = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        handler_with_model.handle(event_init, mock_context)
        
        # Ahora forzar un error interno
        def mock_predict(*args, **kwargs):
            raise RuntimeError("Internal error with sensitive info")
        
        monkeypatch.setattr(handler_with_model._predictor, "predict", mock_predict)
        
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        
        response = handler_with_model.handle(event, mock_context)
        
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["errors"] == ["Internal server error"]
        # No debe contener el mensaje de error interno
        assert "sensitive info" not in json.dumps(body).lower()

    def test_handle_non_numeric_features(self, handler_with_model, mock_context):
        """Test de features con valores no numéricos."""
        event = {
            "body": json.dumps({"features": [5.1, "abc", 1.4, 0.2]})
        }

        response = handler_with_model.handle(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "errors" in body

    def test_handle_empty_body_string(self, handler_with_model, mock_context):
        """Test con body string vacío."""
        event = {"body": ""}

        response = handler_with_model.handle(event, mock_context)

        # Empty string is not valid JSON, should return 500 (generic error)
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["errors"] == ["Internal server error"]

    def test_handle_model_load_failure(self, mock_context, monkeypatch):
        """Test de error interno cuando el modelo no puede cargarse."""
        from ml_lambda import config

        monkeypatch.setattr(config.config, "artifacts_dir", Path("/nonexistent"))
        monkeypatch.setattr(config.config, "model_filename", "missing.joblib")

        handler = LambdaHandler()
        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }

        response = handler.handle(event, mock_context)

        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["errors"] == ["Internal server error"]

    def test_structured_logging_on_success(self, handler_with_model, mock_context, monkeypatch):
        """Test que logs estructurados incluyen request_id, predicción y latencia (Req 6.6)."""
        logged_calls = []
        original_info = handler_with_model._logger.info

        def capture_info(message, **kwargs):
            logged_calls.append({"message": message, **kwargs})
            return original_info(message, **kwargs)

        monkeypatch.setattr(handler_with_model._logger, "info", capture_info)

        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }

        handler_with_model.handle(event, mock_context)

        # Find the inference_complete log entry
        inference_log = next(
            (c for c in logged_calls if c["message"] == "inference_complete"), None
        )
        assert inference_log is not None, "Expected 'inference_complete' log entry"
        assert inference_log["request_id"] == "test-request-123"
        assert "prediction" in inference_log
        assert "latency_ms" in inference_log
        assert inference_log["latency_ms"] > 0

    def test_structured_logging_on_error(self, handler_with_model, mock_context, monkeypatch):
        """Test que errores se registran con contexto (Req 6.6)."""
        logged_errors = []
        original_error = handler_with_model._logger.error

        def capture_error(message, **kwargs):
            logged_errors.append({"message": message, **kwargs})
            return original_error(message, **kwargs)

        monkeypatch.setattr(handler_with_model._logger, "error", capture_error)

        # Force model loaded, then break predictor
        event_init = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        handler_with_model.handle(event_init, mock_context)

        def mock_predict(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(handler_with_model._predictor, "predict", mock_predict)

        event = {
            "body": json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
        }
        handler_with_model.handle(event, mock_context)

        assert len(logged_errors) >= 1
        err_log = logged_errors[0]
        assert err_log["request_id"] == "test-request-123"
        assert "error" in err_log

    def test_error_response_cors_headers(self, handler_with_model, mock_context):
        """Test que respuestas de error también incluyen headers CORS."""
        event = {
            "body": json.dumps({"features": "invalid"})
        }

        response = handler_with_model.handle(event, mock_context)

        assert response["statusCode"] == 400
        headers = response["headers"]
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "Access-Control-Allow-Methods" in headers
