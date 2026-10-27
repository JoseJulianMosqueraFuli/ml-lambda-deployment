# AGENTS.md

Instrucciones para agentes de código en este repositorio. Kiro lo carga siempre como steering: no dupliques su contenido en `.kiro/steering/`.

Clasificador Iris (scikit-learn, RandomForest) servido en AWS Lambda como imagen de contenedor detrás de API Gateway (HTTP API). Es un proyecto de aprendizaje.

## Fuentes de verdad

- Trabajo pendiente y prioridades: [docs/BACKLOG.md](docs/BACKLOG.md).
- Comportamiento real: el código, `infrastructure/main.tf` y `.github/workflows/`. Partes de `README*.md` y `docs/` aún describen el despliegue ZIP + S3 anterior (B-11); si contradicen al código, manda el código.
- `legacy/` es el código original: solo referencia. No lo modifiques ni copies sus patrones.

## Entorno

- Python 3.12, igual que la imagen de Lambda. Evita 3.13+: `numpy` 1.26.4 (lock) no tiene wheels para esas versiones.
- Poetry con `poetry.lock`. Reutiliza `.venv/`; si no existe, créalo con `uv venv --python 3.12` y luego ejecuta `poetry install`.
- Sin Poetry instalado, antepón `uvx` (`uvx poetry install`, `uvx poetry run pytest`).

## Comandos

```bash
poetry run pytest                      # unit + property (Hypothesis); CI exige cobertura >= 80 %
poetry run pytest tests/unit -q        # ciclo rápido
poetry run ruff check src/ tests/
poetry run black --check src/ tests/
poetry run mypy src/
poetry run train                       # entrena y escribe artifacts/model.joblib

# Imagen de Lambda (requiere artifacts/model.joblib); la imagen base trae el emulador RIE
docker build --platform linux/amd64 --provenance=false -f docker/Dockerfile -t ml-iris-predictor .
docker run --rm -p 9000:8080 ml-iris-predictor
curl -s -d '{"body": "{\"features\": [5.1, 3.5, 1.4, 0.2]}"}' \
  http://localhost:9000/2015-03-31/functions/function/invocations

# Terraform: solo validación local (init crea .terraform/, que no se versiona)
terraform -chdir=infrastructure fmt -check
terraform -chdir=infrastructure init -backend=false
terraform -chdir=infrastructure validate
```

## Estructura

```text
src/ml_lambda/
  lambda_function.py    entry point de Lambda (CMD del Dockerfile)
  inference/            handler.py (evento HTTP -> respuesta), validator.py, predictor.py
  data/, training/      carga y escalado; RandomForest con validación cruzada y métricas
  model/serializer.py   modelo + metadatos con joblib
  config.py             configuración central (`config`)
  utils/                StructuredLogger (JSON) y excepciones (base `MLLambdaError`)
  deploy/packager.py    empaquetado ZIP, sin uso desde la migración a contenedor (D-2)
scripts/                train.py (`poetry run train`), lint.py, verify_*.py
tests/                  unit/, property/ (Hypothesis), integration/ (vacío)
docker/Dockerfile       imagen: dependencias del lock + src + artifacts/model.joblib
infrastructure/main.tf  ECR, Lambda (imagen), HTTP API, IAM, CloudWatch
```

## Convenciones

- black y ruff (línea 100), mypy. `X` y `X_train` son válidos (convención de scikit-learn).
- Docstrings estilo Google, en el idioma del archivo que tocas.
- El handler recibe features crudas en cm. Todo preprocesamiento del entrenamiento viaja dentro del artefacto servido (p. ej. `sklearn.pipeline.Pipeline`).
- Errores: usa o extiende `utils/exceptions.py`. `InputValidationError` → 400; cualquier otra → 500 con mensaje genérico, sin detalles internos.
- Logs con `StructuredLogger`, sin `print` en `src/`. Rutas y parámetros desde `ml_lambda.config.config`.
- Tests sin red ni credenciales AWS. Cada bug corregido lleva un test que falla antes del arreglo.
- Commits: `tipo(scope): descripción` (`feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`).

## Forma de trabajo

1. Antes de editar, revisa la rama y `git status --short`; no pises cambios ajenos.
2. Un ítem del backlog por rama (`tipo/descripcion-corta`) y por PR. El trabajo nuevo que encuentres va al backlog, no al ítem actual.
3. Cumple el "Listo cuando" del ítem y la verificación de abajo; actualiza su estado en el mismo PR.
4. Responde en español. Al terminar, indica archivos cambiados, comandos ejecutados con su resultado y lo que no pudiste verificar.

## Verificación

- Python: `pytest`, `ruff`, `black --check` y `mypy` en verde (hoy pasan en `main`).
- Inferencia: además de `statusCode == 200`, comprueba que `[5.1, 3.5, 1.4, 0.2]` devuelve `setosa` (falla hasta cerrar B-01).
- Terraform: `fmt -check` y `validate`.
- Solo documentación: enlaces y rutas válidos y `git diff --check`.

## Límites

- Pregunta antes, salvo que el ítem asignado lo pida: cambiar dependencias (`pyproject.toml`, `poetry.lock`), el contrato de la API (request, response, códigos HTTP), `infrastructure/` o los workflows, o borrar archivos.
- Nunca: `terraform apply`/`destroy`, `docker push` ni comandos `aws` que creen o modifiquen recursos; commitear secretos, `.env`, `*.tfstate`, `.terraform/`, `*.joblib` o ZIP; push directo a `main`; desactivar tests, bajar `--cov-fail-under` o añadir `|| true` para pasar CI.
