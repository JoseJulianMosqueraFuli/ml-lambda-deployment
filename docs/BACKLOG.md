# Backlog

Fuente única del trabajo pendiente del proyecto. Revisión inicial: 2026-10-26 sobre `main` (`a2bf4df`).

## Cómo usar

- El orden es la prioridad: toma el primer ítem pendiente cuyas dependencias y decisiones estén cerradas.
- Un ítem por rama y PR. Al cerrarlo, márcalo `[x]` con el enlace al PR. No renumeres IDs.
- Trabajo nuevo: ítem nuevo al final de su sección, con el siguiente ID libre.
- Estados: `[ ]` pendiente · `[~]` en curso (rama) · `[x]` hecho · `[!]` bloqueado (motivo).

## Estado verificado (2026-10-26)

- Funciona: 115 tests con 95 % de cobertura; ruff, black y mypy sin errores; `poetry run train`; `terraform validate`.
- Falla: el modelo servido acierta 50 de 150 muestras (33 %) con features crudas; `GET /health`, JSON inválido y body vacío devuelven 500.
- Sin verificar: ejecuciones reales de GitHub Actions, `docker build` local, `terraform plan`/`apply` y el despliegue en AWS.

## Decisiones pendientes

- [ ] D-1 Acceso a la API. Hoy es pública, sin autenticación, con CORS `*` y sin throttling. Opciones: (a) pública con throttling y límite de concurrencia; (b) IAM (SigV4); (c) JWT authorizer si habrá frontend. Recomendado: (a) desde el primer despliegue; (b) o (c) si se abre a terceros. Afecta B-04 y B-07.
- [ ] D-2 Ruta ZIP. CI y deploy solo usan la imagen. `src/ml_lambda/deploy/packager.py` instala wheels de la plataforma local y su límite de 50 MB rechaza el paquete real (~74 MB comprimido con wheels Linux). Recomendado: eliminarla. Afecta B-10.
- [ ] D-3 Promoción entre entornos. Hoy hay un repo ECR por entorno y el deploy reentrena el modelo. Recomendado: entrenar y construir una vez y promover el mismo digest. Afecta B-06.

## P0 · Antes de desplegar

- [ ] B-01 Corregir el training/serving skew
  - Problema: `scripts/train.py` entrena con features escaladas (`DataProcessor.normalize`) pero solo serializa el RandomForest, y el handler predice con features crudas: todo sale `virginica`, incluido `[5.1, 3.5, 1.4, 0.2]` (setosa). Los tests no lo detectan porque `trained_model` (conftest) entrena sin escalar y los smoke tests solo miran `statusCode`.
  - Hacer: serializar un `sklearn.pipeline.Pipeline` (scaler + modelo) o quitar el escalado (RandomForest no lo necesita).
  - Listo cuando: un test en `tests/integration/` entrena con el flujo real y, vía `LambdaHandler` con features crudas, predice 0, 1 y 2 para las muestras 0, 50 y 100 de Iris y acierta al menos el 90 % del dataset; los smoke tests de `ci.yml` y `deploy.yml` exigen `class_name == "setosa"`.
- [ ] B-02 Ignorar artefactos de Terraform
  - Problema: el state es local y `.gitignore` no cubre `.terraform/` ni `*.tfstate*`, que pueden contener datos sensibles.
  - Hacer: ignorar `.terraform/`, `*.tfstate`, `*.tfstate.*` y `crash.log`; versionar `.terraform.lock.hcl`.
  - Listo cuando: tras `terraform init` en `infrastructure/`, `git status` solo muestra `.terraform.lock.hcl`.
- [ ] B-03 Health check y errores de cliente en el handler (cambia el contrato de la API)
  - Problema: Terraform publica `GET /health`, pero con el payload 1.0 (default del provider) llega `body: null` y el handler responde 500. JSON inválido o body vacío también dan 500; `test_handle_invalid_json` y `test_handle_empty_body_string` fijan ese comportamiento.
  - Hacer: enrutar por método y ruta (1.0: `httpMethod`/`path`; 2.0: `requestContext.http.method`/`rawPath`); `GET /health` → 200 con la versión del modelo; ruta desconocida → 404; JSON inválido, vacío o que no sea un objeto → 400. Dejar una sola instancia de `LambdaHandler` (hoy hay una en `handler.py` y otra en `lambda_function.py`).
  - Listo cuando: hay tests unitarios para ambos formatos de payload y cada código de respuesta, y la sección API del README está actualizada.

## P1 · Despliegue de punta a punta

- [ ] B-04 Ajustes de Terraform antes del primer `apply`
  - Fijar `payload_format_version` en la integración, alineado con B-03.
  - Crear el log group de Lambda antes que la función (nombre desde variables o `logging_config`) para no chocar con el que crea Lambda al invocarse.
  - Agregar `alarm_actions` (SNS + email): hoy las alarmas no notifican a nadie.
  - Throttling del stage y límite de concurrencia según D-1.
  - Listo cuando: `fmt -check` y `validate` pasan y el `plan` está revisado.
- [ ] B-05 Bootstrap del primer despliegue
  - Problema: el primer `apply` crea el repo ECR vacío y, en el mismo plan, la Lambda con `<repo>:latest`, que no existe, así que falla. El repo es `IMMUTABLE` y los workflows publican tags por SHA, nunca `latest`.
  - Hacer: documentar y automatizar el arranque (p. ej. `apply -target=aws_ecr_repository.predictor`, push con tag SHA y `apply -var image_tag=<sha>`) y quitar el default `latest`.
  - Listo cuando: en una cuenta vacía, los pasos de `docs/DEPLOYMENT.md` dejan `POST /predict` y `GET /health` respondiendo vía API Gateway.
  - Depende de: B-02, B-03, B-04.
- [ ] B-06 Pipeline de deploy confiable
  - Problema: `deploy.yml` reentrena y reconstruye, así que lo desplegado no es lo que probó CI; usa access keys de larga duración; el rollback documentado usa un alias `live` que no existe, y la lifecycle policy de ECR borra las imágenes más allá de las 10 últimas.
  - Hacer: promover el artefacto probado (D-3); OIDC (`role-to-assume` con `id-token: write`) y el rol en Terraform; smoke test también con `curl` a la URL de API Gateway; un rollback que funcione.
  - Listo cuando: un deploy a staging desde Actions pasa los smoke tests sin secrets `AWS_*` de larga duración.
  - Depende de: B-05, D-3.

## P2 · Endurecimiento

- [ ] B-07 Autenticación de la API según D-1, con `allow_origins` restringido si hay frontend.
- [ ] B-08 Dependencias y tooling
  - Sacar `boto3` de las dependencias de producción: `src/` no lo usa y la imagen base ya lo incluye.
  - `python = "^3.11"` admite 3.13+, donde `numpy` 1.26.4 no instala desde wheels: acotar a `<3.13` o subir a numpy 2.x.
  - Fijar la versión de Poetry en CI (hoy `latest`) y alinearla con la del Dockerfile.
- [ ] B-09 Estado remoto de Terraform (backend S3 con `use_lockfile = true`) si CI va a aplicar Terraform o trabaja más de una persona.

## P3 · Limpieza y documentación

- [ ] B-10 Limpieza
  - Ruta ZIP, según D-2: `src/ml_lambda/deploy/packager.py`, `scripts/package.py`, `scripts/export_requirements.py`, `PackageTooLargeError` y `tests/unit/test_packager.py`.
  - Sin uso: `AWSCredentialsError` y `DeploymentError`; en `config`, `max_body_size` (duplica `MAX_BODY_SIZE` del validator), `log_level` (Terraform define `LOG_LEVEL` y nadie lo lee), `lambda_memory` (256; Terraform usa 512) y `metadata_*`.
  - Decidir si se quedan `scripts/verify_*.py` (checkpoints del spec) y si se versiona `.kiro/` (`.gitignore` lo ignora, pero su comentario dice "mantener en repo").
- [ ] B-11 Documentación alineada con el código (después de P0 y P1)
  - `README.md`: "Current Status" y la estructura están desactualizados (reemplazar el estado por un enlace a este backlog) y la salida de ejemplo de `train` no es real.
  - `README.md` y `README.es.md`: `poetry run lint` no existe (registrar `scripts.lint:main` o quitarlo); prometen verificación de integridad que `ModelSerializer.load()` no aplica.
  - `docs/ARCHITECTURE.md`: flujo ZIP + S3, `validator.validate()`/`is_valid` (ya no existen) y "Próximos pasos" ya cerrados.
  - `docs/DEPLOYMENT.md`: todo el flujo ZIP + S3, el alias `live` y costos sin ECR.
  - Al cerrar, quitar el aviso de "Fuentes de verdad" en `AGENTS.md`.
