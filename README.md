# ML Lambda Deployment 🚀

Proyecto de aprendizaje para desplegar un modelo de Machine Learning en AWS Lambda con API Gateway.

## 📋 Descripción

Este proyecto implementa un flujo completo de MLOps básico:

1. **Entrenamiento local** de un modelo de clasificación (Iris dataset)
2. **Serialización** del modelo entrenado
3. **Empaquetado** para AWS Lambda
4. **Despliegue** como API serverless

## 🎯 Objetivos de Aprendizaje

- Gestión de dependencias con Poetry
- Entrenamiento y evaluación de modelos con scikit-learn
- Serialización de modelos ML
- Despliegue serverless en AWS Lambda
- Configuración de API Gateway
- Testing con pytest y property-based testing (Hypothesis)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE LOCAL                                │
│  Dataset Iris → Entrenamiento → Serialización → Empaquetado │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE CLOUD (AWS)                          │
│  Cliente HTTP → API Gateway → Lambda → Modelo → Predicción  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Requisitos Previos

- Python 3.11+
- Poetry
- AWS CLI configurado (para despliegue)

### Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd ml-lambda-deployment

# Instalar dependencias con Poetry
poetry install

# Activar entorno virtual
poetry shell
```

### Entrenamiento Local

```bash
# Entrenar modelo
poetry run train

# Ejecutar tests
poetry run test
```

### Despliegue a AWS

```bash
# Crear paquete de despliegue
poetry run python scripts/package.py

# Desplegar a AWS
poetry run python scripts/deploy.py --environment dev
```

## 📁 Estructura del Proyecto

```
ml-lambda-deployment/
├── src/ml_lambda/          # Código fuente principal
│   ├── data/               # Procesamiento de datos
│   ├── training/           # Entrenamiento y evaluación
│   ├── model/              # Serialización de modelos
│   ├── inference/          # Handler Lambda y validación
│   ├── utils/              # Logging y excepciones
│   └── deploy/             # Empaquetado y despliegue
├── tests/                  # Tests unitarios, integración y property
├── scripts/                # Scripts de entrenamiento y despliegue
├── artifacts/              # Modelos serializados
└── legacy/                 # Código original (referencia)
```

## 🔌 API

### POST /predict

Realiza una predicción de clasificación de flores Iris.

**Request:**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [0.95, 0.03, 0.02],
  "latency_ms": 12.5
}
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
poetry run pytest

# Con cobertura
poetry run pytest --cov=src/ml_lambda

# Solo property tests
poetry run pytest tests/property/
```

## 📚 Documentación Adicional

- [Especificación de Requisitos](.kiro/specs/ml-lambda-deployment/requirements.md)
- [Documento de Diseño](.kiro/specs/ml-lambda-deployment/design.md)
- [Plan de Implementación](.kiro/specs/ml-lambda-deployment/tasks.md)

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE)

## 👤 Autor

Proyecto de aprendizaje - ML + AWS Lambda
