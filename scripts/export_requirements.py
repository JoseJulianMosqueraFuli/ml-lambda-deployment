"""Script para exportar requirements.txt sin dependencias de desarrollo."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Exporta requirements.txt para Lambda (sin deps de desarrollo)."""
    output_path = Path("requirements.txt")
    
    try:
        result = subprocess.run(
            ["poetry", "export", "--without", "dev", "--format", "requirements.txt"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        output_path.write_text(result.stdout)
        print(f"✓ Requirements exportados a {output_path}")
        print(f"  Líneas: {len(result.stdout.splitlines())}")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error exportando requirements: {e.stderr}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
