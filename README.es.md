# ML Lambda Deployment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | Español

Despliega modelos de machine learning en AWS Lambda con API Gateway. Un pipeline MLOps completo desde entrenamiento hasta inferencia serverless.

## Descripción

Este proyecto implementa un flujo de despliegue ML de extremo a extremo:

- **Procesamiento de Datos**: Carga, validación, división y normalización del dataset Iris
- **Entrenamiento de Modelo**: Entrena un clasificador Random Forest con validación cruzada y métricas completas
- **Serialización de Modelo**: Guarda modelos entrenados con metadatos y verificación de integridad
- **Inferencia Serverless**: Despliega como función AWS Lambda detrás de API Gateway
- **Logging Estructurado**: Logs en formato JSON para observabilidad

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE LOCAL                             │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  Dataset │───▶│ Procesar │───▶│ Entrenar │───▶│Serializar│  │
│   │   Iris   │    │ y Dividir│    │  Modelo  │    │  Modelo  │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DESPLIEGUE AWS                             │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │ Cliente  │───▶│   API    │───▶│  Lambda  │───▶│  Modelo  │  │
│   │   HTTP   │    │ Gateway  │    │ Handler  │    │ Predecir │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- AWS CLI configurado (para despliegue)

## Instalación

```bash
git clone https://github.com/JoseJulianMosqueraFuli/ml-lambda-deployment.git
cd ml-lambda-deployment
poetry install
```

## Uso

### Entrenar un Modelo

```bash
poetry run train
```

Esto realizará:

1. Cargar y preprocesar el dataset Iris (división 80/20 train/test)
2. Entrenar un clasificador Random Forest con validación cruzada
3. Evaluar métricas (accuracy, precision, recall, F1-score)
4. Guardar el modelo en `artifacts/`

### Ejecutar Tests

```bash
# Todos los tests
poetry run pytest

# Con reporte de cobertura
poetry run pytest --cov=src/ml_lambda --cov-report=term-missing

# Solo tests basados en propiedades
poetry run pytest tests/property/
```

### Calidad de Código

```bash
poetry run lint
```

## Estructura del Proyecto

```
ml-lambda-deployment/
├── src/ml_lambda/
│   ├── config.py           # Dataclasses de configuración
│   ├── data/               # Carga y preprocesamiento de datos
│   ├── training/           # Entrenamiento y evaluación de modelos
│   ├── model/              # Serialización de modelos
│   ├── inference/          # Handler Lambda y validación
│   ├── deploy/             # Empaquetado y despliegue AWS
│   └── utils/              # Logging y excepciones personalizadas
├── tests/
│   ├── unit/               # Tests unitarios
│   ├── property/           # Tests basados en propiedades (Hypothesis)
│   └── integration/        # Tests de integración
├── scripts/                # Scripts CLI
├── artifacts/              # Modelos entrenados
└── docs/                   # Documentación
```

## Referencia de API

### POST /predict

Clasifica una flor Iris basándose en sus características.

**Request**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Características (en orden): longitud del sépalo, ancho del sépalo, longitud del pétalo, ancho del pétalo (cm)

**Response**

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [0.95, 0.03, 0.02],
  "latency_ms": 12.5
}
```

## Documentación

- [Guía de Arquitectura](docs/ARCHITECTURE.md) - Diseño del sistema y flujo de datos
- [Conceptos de ML](docs/CONCEPTS.md) - Fundamentos de machine learning

## Estrategia de Testing

El proyecto utiliza un enfoque de testing integral:

- **Tests Unitarios**: Validan componentes individuales
- **Tests Basados en Propiedades**: Usan [Hypothesis](https://hypothesis.readthedocs.io/) para verificar invariantes con inputs aleatorios
- **Tests de Integración**: Verifican flujos de trabajo de extremo a extremo

## Licencia

[MIT](LICENSE)
