"""Despliegue a AWS Lambda y API Gateway."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DeploymentResult:
    """Resultado del despliegue."""

    function_arn: str
    api_endpoint: str
    version: str
    deployed_at: datetime


class AWSDeployer:
    """Despliega a AWS Lambda y API Gateway."""

    def __init__(self, environment: str = "dev"):
        self.environment = environment
        self._lambda_client = None
        self._apigateway_client = None

    def validate_credentials(self) -> bool:
        """Valida credenciales AWS."""
        # TODO: Implementar en tarea 19.1
        raise NotImplementedError

    def deploy_lambda(self, package_path: Path, function_name: str) -> str:
        """Despliega o actualiza función Lambda. Retorna ARN."""
        # TODO: Implementar en tarea 19.2
        raise NotImplementedError

    def setup_api_gateway(self, function_arn: str, api_name: str) -> str:
        """Configura API Gateway. Retorna URL del endpoint."""
        # TODO: Implementar en tarea 19.3
        raise NotImplementedError

    def deploy(self, package_path: Path, function_name: str) -> DeploymentResult:
        """Despliegue completo: Lambda + API Gateway."""
        # TODO: Implementar en tarea 19.4
        raise NotImplementedError

    def rollback(self, function_name: str, version: str) -> None:
        """Rollback a versión anterior."""
        # TODO: Implementar en tarea 19.4
        raise NotImplementedError
