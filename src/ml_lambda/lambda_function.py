"""Entry point para AWS Lambda.

Este módulo expone la función lambda_handler que AWS Lambda invocará
para procesar solicitudes de inferencia.
"""

from ml_lambda.inference.handler import LambdaHandler

# Instancia global para reutilizar entre invocaciones (warm starts)
_handler = LambdaHandler()


def lambda_handler(event, context):
    """Entry point de AWS Lambda.
    
    Args:
        event: Evento de API Gateway con el body de la solicitud
        context: Contexto de Lambda con información de la invocación
        
    Returns:
        Respuesta HTTP con predicción o error
    """
    return _handler.handle(event, context)
