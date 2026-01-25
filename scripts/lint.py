"""Script de linting del proyecto."""

import subprocess
import sys


def main() -> int:
    """Ejecuta black, ruff y mypy."""
    commands = [
        (["black", "--check", "src/", "tests/", "scripts/"], "Black"),
        (["ruff", "check", "src/", "tests/", "scripts/"], "Ruff"),
        (["mypy", "src/"], "Mypy"),
    ]
    
    failed = False
    for cmd, name in commands:
        print(f"\n{'='*40}")
        print(f"Ejecutando {name}...")
        print(f"{'='*40}")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed = True
            print(f"✗ {name} encontró problemas")
        else:
            print(f"✓ {name} OK")
    
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
