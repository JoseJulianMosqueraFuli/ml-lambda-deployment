"""Tests unitarios para la creación de paquetes Lambda."""

import hashlib
import zipfile
from pathlib import Path

import pytest
from ml_lambda.deploy import packager as packager_module
from ml_lambda.deploy.packager import PackageBuilder
from ml_lambda.utils.exceptions import PackageTooLargeError


def _build_without_dependencies(
    builder: PackageBuilder, source_dir: Path, model_path: Path, output_path: Path
):
    """Construye un paquete sin descargar dependencias durante los tests."""
    builder._install_dependencies = lambda target_dir: None
    return builder.build(source_dir, model_path, output_path)


class TestPackageBuilder:
    """Tests para PackageBuilder."""

    def test_build_includes_source_dependencies_and_model(self, tmp_path):
        source_dir = tmp_path / "src"
        (source_dir / "ml_lambda").mkdir(parents=True)
        (source_dir / "ml_lambda" / "handler.py").write_text("handler = True")
        (source_dir / "tests").mkdir()
        (source_dir / "tests" / "test_handler.py").write_text("test = True")
        (source_dir / "ml_lambda" / "__pycache__").mkdir()
        (source_dir / "ml_lambda" / "__pycache__" / "handler.pyc").write_bytes(b"cache")
        (source_dir / "ml_lambda" / "handler.pyc").write_bytes(b"bytecode")

        model_path = tmp_path / "trained-model.joblib"
        model_path.write_bytes(b"serialized model")
        output_path = tmp_path / "build" / "deployment.zip"

        package_info = _build_without_dependencies(
            PackageBuilder(), source_dir, model_path, output_path
        )

        with zipfile.ZipFile(output_path) as package:
            names = set(package.namelist())

        assert names == {"ml_lambda/handler.py", "artifacts/model.joblib"}
        assert package_info.included_files == sorted(names)
        assert package_info.path == output_path
        assert package_info.size_bytes == output_path.stat().st_size
        assert package_info.sha256_hash == hashlib.sha256(output_path.read_bytes()).hexdigest()

    def test_build_rejects_packages_over_size_limit(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "payload.bin").write_bytes(bytes(range(256)) * 100)
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"model")
        output_path = tmp_path / "deployment.zip"

        builder = PackageBuilder()
        builder.MAX_SIZE_MB = 0.0001

        with pytest.raises(PackageTooLargeError):
            _build_without_dependencies(builder, source_dir, model_path, output_path)

        assert not output_path.exists()

    def test_build_requires_source_and_model(self, tmp_path):
        builder = PackageBuilder()
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"model")

        with pytest.raises(FileNotFoundError, match="Source directory"):
            builder.build(tmp_path / "missing", model_path, tmp_path / "package.zip")

        source_dir = tmp_path / "src"
        source_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Model file"):
            builder.build(source_dir, tmp_path / "missing.joblib", tmp_path / "package.zip")

    def test_production_requirements_fall_back_when_export_is_unavailable(self, monkeypatch):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[1] == "export":
                return packager_module.subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="export unavailable"
                )
            return packager_module.subprocess.CompletedProcess(
                command,
                0,
                stdout='[{"name": "numpy", "version": "1.26.4"}]',
                stderr="",
            )

        monkeypatch.setattr(packager_module.subprocess, "run", fake_run)

        requirements = packager_module.get_production_requirements()

        assert requirements == "numpy==1.26.4"
        assert calls[0][1] == "export"
        assert calls[1][1] == "show"
