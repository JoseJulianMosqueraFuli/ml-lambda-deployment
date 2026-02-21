"""Script de entrenamiento del modelo."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ml_lambda.config import config
from ml_lambda.data.processor import DataProcessor
from ml_lambda.model.serializer import ModelMetadata, ModelSerializer
from ml_lambda.training.evaluator import ModelEvaluator
from ml_lambda.training.trainer import ModelTrainer, TrainingConfig
from ml_lambda.utils.logging import StructuredLogger


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenar modelo de clasificación Iris")
    parser.add_argument("--n-estimators", type=int, default=config.n_estimators, help="Número de árboles en el Random Forest")
    parser.add_argument("--max-depth", type=int, default=config.max_depth, help="Profundidad máxima de los árboles")
    parser.add_argument("--min-samples-split", type=int, default=config.min_samples_split, help="Mínimo de muestras para dividir un nodo")
    parser.add_argument("--output-dir", type=Path, default=config.artifacts_dir, help="Directorio de salida para el modelo")
    parser.add_argument("--random-state", type=int, default=config.random_state, help="Semilla aleatoria para reproducibilidad")
    return parser.parse_args()


def main() -> int:
    """Punto de entrada para entrenamiento."""
    args = parse_args()
    logger = StructuredLogger("train")
    logger.info("Iniciando pipeline de entrenamiento")

    # 1. Cargar y procesar datos
    processor = DataProcessor(test_size=config.test_size, random_state=args.random_state)
    X, y = processor.load_iris()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train_norm = processor.normalize(X_train, fit=True)
    X_test_norm = processor.normalize(X_test, fit=False)
    logger.info("Datos cargados y procesados", extra={"train_size": len(X_train), "test_size": len(X_test)})

    # 2. Entrenar modelo
    training_config = TrainingConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_cv_folds=config.n_cv_folds,
    )
    trainer = ModelTrainer(training_config)
    result = trainer.train(X_train_norm, y_train)
    logger.info("Modelo entrenado", extra={"cv_mean": result.cv_mean, "cv_std": result.cv_std, "training_time": result.training_time_seconds})

    # 3. Evaluar modelo
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(result.model, X_test_norm, y_test)
    evaluator.check_accuracy_threshold(metrics, config.accuracy_threshold)
    logger.info("Modelo evaluado", extra={"accuracy": metrics.accuracy, "f1_score": metrics.f1_score})

    # 4. Guardar modelo
    output_path = args.output_dir / config.model_filename
    metadata = ModelMetadata(
        version=config.version,
        created_at=datetime.now(),
        accuracy=metrics.accuracy,
        n_features=config.expected_features,
        n_classes=len(config.class_names),
        feature_names=config.feature_names,
        class_names=config.class_names,
        training_config={"n_estimators": training_config.n_estimators, "max_depth": training_config.max_depth, "min_samples_split": training_config.min_samples_split, "random_state": training_config.random_state, "n_cv_folds": training_config.n_cv_folds},
    )
    serializer = ModelSerializer()
    model_hash = serializer.save(result.model, metadata, output_path)
    logger.info("Modelo guardado", extra={"path": str(output_path), "hash": model_hash})

    return 0


if __name__ == "__main__":
    sys.exit(main())
