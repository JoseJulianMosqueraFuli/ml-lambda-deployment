"""Excepciones personalizadas del proyecto."""


class MLLambdaError(Exception):
    """Excepción base del proyecto."""

    pass


class DataValidationError(MLLambdaError):
    """Error en validación de datos.
    
    Se lanza cuando los datos de entrada no cumplen con los requisitos:
    - Valores nulos detectados
    - Formato incorrecto
    - Tipos de datos inválidos
    """

    pass


class ModelNotTrainedError(MLLambdaError):
    """Modelo no ha sido entrenado.
    
    Se lanza cuando se intenta usar un modelo que no ha sido entrenado.
    """

    pass


class ModelNotFoundError(MLLambdaError):
    """Archivo de modelo no encontrado.
    
    Se lanza cuando el archivo del modelo serializado no existe.
    """

    pass


class ModelCorruptedError(MLLambdaError):
    """Archivo de modelo corrupto.
    
    Se lanza cuando el archivo del modelo no puede ser deserializado
    o falla la validación de integridad.
    """

    pass


class InputValidationError(MLLambdaError):
    """Error en validación de entrada de API.
    
    Se lanza cuando la entrada de la API no cumple con el formato esperado:
    - Features no es una lista
    - Número incorrecto de features
    - Tipos de datos incorrectos
    """

    pass


class PackageTooLargeError(MLLambdaError):
    """Paquete excede límite de tamaño.
    
    Se lanza cuando el paquete ZIP excede el límite de 50MB para Lambda.
    """

    pass


class AWSCredentialsError(MLLambdaError):
    """Credenciales AWS inválidas o no configuradas.
    
    Se lanza cuando no se pueden validar las credenciales de AWS.
    """

    pass


class DeploymentError(MLLambdaError):
    """Error durante despliegue.
    
    Se lanza cuando ocurre un error durante el despliegue a AWS.
    """

    pass
