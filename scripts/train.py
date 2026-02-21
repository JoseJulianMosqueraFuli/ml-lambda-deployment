"""Script de entrenamiento del modelo."""

import sys
from datetime import datetime

from ml_lambda.config import config
from ml_lambda.data.processor import DataProcessor
from ml_lambda.model.serializer import ModelMetadata, ModelSerializer
from ml_lambda.training.evaluator import ModelEvaluator
from ml_lambda.training.trainer import ModelTrainer, TrainingConfig
from ml_lambda.utils.logging import StructuredLogger


def main() -> int:
    """Punto de entrada para entrenamiento."""
    logger = StructuredLogger("train")
    logger.info("Iniciando pipeline de entrenamiento")

    # 1. Cargar y procesar datos
    processor = DataProcessor(test_size=config.test_size, random_state=config.random_state)
    X, y = processor.load_iris()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train_norm = processor.normalize(X_train, fit=True)
    X_test_norm = processor.normalize(X_test, fit=False)
    logger.info("Datos cargados y procesados", extra={"train_size": len(X_train), "test_size": len(X_test)})

    # 2. Entrenar modelo
    training_config = TrainingConfig(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state,
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
    model_hash = serializer.save(result.model, metadata, config.model_path)
    logger.info("Modelo guardado", extra={"path": str(config.model_path), "hash": model_hash})

    return 0


if __name__ == "__main__":
    sys.exit(main())
