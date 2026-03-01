"""Handler principal de AWS Lambda."""

import json
import time
from pathlib import Path
from typing import Any

from ..config import config
from ..model.serializer import ModelSerializer
from ..inference.validator import InputValidator
from ..inference.predictor import Predictor
from ..utils.logging import StructuredLogger


class LambdaHandler:
    """Handler para AWS Lambda."""

    def __init__(self):
        self._model = None
        self._metadata = None
        self._predictor = None
        self._validator = InputValidator()
        self._logger = StructuredLogger("lambda_handler")

    def _load_model_once(self) -> None:
        """Carga el modelo una sola vez (cold start)."""
        if self._model is None:
            self._logger.info("Loading model (cold start)")
            serializer = ModelSerializer()
            result = serializer.load(config.model_path)
            self._model = result.model
            self._metadata = result.metadata
            self._predictor = Predictor(self._model, self._metadata.class_names)
            self._logger.info("Model loaded successfully")

    def handle(self, event: dict[str, Any], context: Any) -> dict[str, Any]:
        """Procesa solicitud de inferencia.
        
        Args:
            event: Evento de API Gateway
            context: Contexto de Lambda
            
        Returns:
            Respuesta HTTP con predicción o error
        """
        request_id = context.aws_request_id if context and hasattr(context, 'aws_request_id') else "local"
        start_time = time.perf_counter()
        
        try:
            # Cargar modelo en cold start
            self._load_model_once()
            
            # Parsear body
            body = self._parse_body(event)
            
            # Validar entrada
            if "features" not in body:
                return self._error_response(400, ["Missing 'features' field in request body"])
            
            features = self._validator.validate_features(body["features"])
            
            # Realizar predicción
            result = self._predictor.predict(features)
            
            # Calcular latencia
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Log estructurado
            self._logger.info(
                "inference_complete",
                request_id=request_id,
                prediction=result.prediction,
                latency_ms=round(latency_ms, 2)
            )
            
            return self._success_response({
                "prediction": result.prediction,
                "class_name": result.class_name,
                "probabilities": result.probabilities,
                "latency_ms": round(latency_ms, 2)
            })
            
        except Exception as e:
            self._logger.error(
                "inference_error",
                error=str(e),
                error_type=type(e).__name__,
                request_id=request_id
            )
            
            # Determinar código de error
            from ..utils.exceptions import InputValidationError
            if isinstance(e, InputValidationError):
                return self._error_response(400, [str(e)])
            else:
                return self._error_response(500, ["Internal server error"])

    def _parse_body(self, event: dict) -> dict:
        """Parsea el body del evento.
        
        Args:
            event: Evento de Lambda (puede venir de API Gateway o directo)
            
        Returns:
            Body parseado como diccionario
            
        Raises:
            Exception: Si el JSON es inválido
        """
        body = event.get("body", event)
        
        # Si body es string, parsear como JSON
        if isinstance(body, str):
            # Validar tamaño antes de parsear
            self._validator.validate_body_size(body)
            try:
                return json.loads(body)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in request body: {str(e)}")
        
        # Si ya es dict, retornar directamente
        return body

    def _success_response(self, data: dict) -> dict:
        """Construye respuesta exitosa.
        
        Args:
            data: Datos a incluir en el body de la respuesta
            
        Returns:
            Respuesta HTTP con código 200
        """
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps(data)
        }

    def _error_response(self, code: int, errors: list[str]) -> dict:
        """Construye respuesta de error.
        
        Args:
            code: Código HTTP de error (400, 500, etc.)
            errors: Lista de mensajes de error
            
        Returns:
            Respuesta HTTP con código de error
        """
        return {
            "statusCode": code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"errors": errors})
        }


# Instancia global para reutilizar entre invocaciones
_handler = LambdaHandler()


def lambda_handler(event, context):
    """Entry point de Lambda."""
    return _handler.handle(event, context)
