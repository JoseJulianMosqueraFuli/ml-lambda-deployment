"""Construcción de paquetes de despliegue para Lambda."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackageInfo:
    """Información del paquete creado."""

    path: Path
    size_bytes: int
    size_mb: float
    sha256_hash: str
    included_files: list[str]


class PackageBuilder:
    """Construye paquete de despliegue para Lambda."""

    MAX_SIZE_MB = 50

    EXCLUDE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "tests/",
        ".git/",
        "*.egg-info/",
    ]

    def build(
        self, source_dir: Path, model_path: Path, output_path: Path
    ) -> PackageInfo:
        """Construye el paquete ZIP."""
        # TODO: Implementar en tarea 17.3
        raise NotImplementedError

    def _install_dependencies(self, target_dir: Path) -> None:
        """Instala dependencias de producción."""
        # TODO: Implementar en tarea 17.1
        raise NotImplementedError

    def _compute_hash(self, path: Path) -> str:
        """Calcula SHA256 del archivo."""
        # TODO: Implementar en tarea 17.4
        raise NotImplementedError
