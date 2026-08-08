"""Script para exportar requirements.txt sin dependencias de desarrollo."""

import sys
from pathlib import Path

from ml_lambda.deploy.packager import get_production_requirements


def main() -> int:
    """Exporta requirements.txt para Lambda (sin deps de desarrollo)."""
    output_path = Path("requirements.txt")

    try:
        requirements = get_production_requirements()
        output_path.write_text(requirements)
        print(f"✓ Requirements exportados a {output_path}")
        print(f"  Líneas: {len(requirements.splitlines())}")
        return 0

    except RuntimeError as error:
        print(f"✗ Error exportando requirements: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
