# Guía de Contribución

## Flujo de Desarrollo

1. Crear branch desde `main`
2. Implementar cambios siguiendo el spec
3. Ejecutar tests localmente
4. Crear Pull Request

## Estándares de Código

- Formateo con `black`
- Linting con `ruff`
- Type hints con `mypy`
- Docstrings estilo Google

## Ejecutar Validaciones

```bash
# Formatear código
poetry run black src/ tests/

# Linting
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/

# Tests
poetry run pytest
```

## Estructura de Commits

```
tipo(scope): descripción breve

- feat: nueva funcionalidad
- fix: corrección de bug
- docs: documentación
- test: tests
- refactor: refactorización
```
