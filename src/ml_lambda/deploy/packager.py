"""Construcción de paquetes de despliegue para Lambda."""

import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import config
from ..utils.exceptions import PackageTooLargeError


def get_production_requirements() -> str:
    """Obtiene las dependencias de producción en formato requirements."""
    export_command = [
        "poetry",
        "export",
        "--without",
        "dev",
        "--format",
        "requirements.txt",
    ]
    export_result = subprocess.run(
        export_command,
        capture_output=True,
        text=True,
    )
    if export_result.returncode == 0:
        return export_result.stdout

    show_result = subprocess.run(
        ["poetry", "show", "--only", "main", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if show_result.returncode != 0:
        error = show_result.stderr.strip() or export_result.stderr.strip()
        raise RuntimeError(f"Unable to resolve production dependencies: {error}")

    packages = json.loads(show_result.stdout)
    return "\n".join(f"{package['name']}=={package['version']}" for package in packages)


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

    def build(self, source_dir: Path, model_path: Path, output_path: Path) -> PackageInfo:
        """Construye el paquete ZIP."""
        source_dir = Path(source_dir)
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ml-lambda-package-") as temp_dir:
            staging_dir = Path(temp_dir)
            self._install_dependencies(staging_dir)
            self._copy_tree(source_dir, staging_dir)

            # The Lambda handler loads the model from artifacts/model.joblib.
            model_destination = staging_dir / "artifacts" / config.model_filename
            model_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, model_destination)

            included_files = self._create_zip(staging_dir, output_path)

        size_bytes = output_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.MAX_SIZE_MB:
            output_path.unlink(missing_ok=True)
            raise PackageTooLargeError(
                f"Deployment package exceeds {self.MAX_SIZE_MB} MB: {size_mb:.2f} MB"
            )

        return PackageInfo(
            path=output_path,
            size_bytes=size_bytes,
            size_mb=size_mb,
            sha256_hash=self._compute_hash(output_path),
            included_files=included_files,
        )

    def _install_dependencies(self, target_dir: Path) -> None:
        """Instala dependencias de producción."""
        target_dir.mkdir(parents=True, exist_ok=True)

        requirements_text = self._production_requirements()

        with tempfile.NamedTemporaryFile(mode="w", suffix="-requirements.txt") as requirements:
            requirements.write(requirements_text)
            requirements.flush()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(target_dir),
                    "--no-compile",
                    "--no-cache-dir",
                    "--requirement",
                    requirements.name,
                ],
                check=True,
            )

    def _production_requirements(self) -> str:
        """Obtiene las dependencias de producción en formato requirements."""
        return get_production_requirements()

    def _compute_hash(self, path: Path) -> str:
        """Calcula SHA256 del archivo."""
        sha256_hash = hashlib.sha256()
        with path.open("rb") as package_file:
            for chunk in iter(lambda: package_file.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _copy_tree(self, source_dir: Path, target_dir: Path) -> None:
        """Copia el código fuente respetando las exclusiones del paquete."""
        for source_path in source_dir.rglob("*"):
            relative_path = source_path.relative_to(source_dir)
            if self._is_excluded(relative_path):
                continue

            target_path = target_dir / relative_path
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

    def _create_zip(self, staging_dir: Path, output_path: Path) -> list[str]:
        """Crea el ZIP y retorna sus archivos en orden determinista."""
        included_files = sorted(
            path.relative_to(staging_dir).as_posix()
            for path in staging_dir.rglob("*")
            if path.is_file() and not self._is_excluded(path.relative_to(staging_dir))
        )

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for relative_path in included_files:
                package.write(staging_dir / relative_path, arcname=relative_path)

        return included_files

    def _is_excluded(self, relative_path: Path) -> bool:
        """Indica si una ruta debe quedar fuera del paquete."""
        parts = relative_path.parts
        for pattern in self.EXCLUDE_PATTERNS:
            normalized_pattern = pattern.rstrip("/")
            if pattern.endswith("/") or normalized_pattern == "__pycache__":
                if any(fnmatch.fnmatch(part, normalized_pattern) for part in parts):
                    return True
            elif fnmatch.fnmatch(relative_path.name, normalized_pattern):
                return True
        return False
