"""Crea un paquete ZIP para AWS Lambda."""

import argparse
import sys
from pathlib import Path

from ml_lambda.config import config
from ml_lambda.deploy.packager import PackageBuilder
from ml_lambda.utils.exceptions import PackageTooLargeError


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos del empaquetado."""
    parser = argparse.ArgumentParser(description="Crear paquete de despliegue para Lambda")
    parser.add_argument("--source-dir", type=Path, default=Path("src"))
    parser.add_argument("--model-path", type=Path, default=config.model_path)
    parser.add_argument("--output-path", type=Path, default=Path("deployment_package.zip"))
    return parser.parse_args()


def main() -> int:
    """Construye el paquete y muestra sus metadatos."""
    args = parse_args()
    try:
        package_info = PackageBuilder().build(
            source_dir=args.source_dir,
            model_path=args.model_path,
            output_path=args.output_path,
        )
    except (FileNotFoundError, PackageTooLargeError, RuntimeError) as error:
        print(f"Package creation failed: {error}", file=sys.stderr)
        return 1

    print(f"Package created: {package_info.path}")
    print(f"Size: {package_info.size_mb:.2f} MB")
    print(f"SHA256: {package_info.sha256_hash}")
    print(f"Included files: {len(package_info.included_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
