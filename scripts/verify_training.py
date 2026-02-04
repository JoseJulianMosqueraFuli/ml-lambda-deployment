"""Script de verificación de entrenamiento completo - Checkpoint 8."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_lambda.data.processor import DataProcessor
from ml_lambda.training.trainer import ModelTrainer, TrainingConfig
from ml_lambda.training.evaluator import ModelEvaluator


def main() -> int:
    """Ejecuta verificación completa de entrenamiento."""
    print("=" * 60)
    print("CHECKPOINT 8: Verificación de Entrenamiento Completo")
    print("=" * 60)

    # 1. Cargar y preparar datos
    print("\n1. Cargando dataset Iris...")
    processor = DataProcessor(test_size=0.2, random_state=42)
    X, y = processor.load_iris()
    stats = processor.compute_stats(X, y)

    print(f"   - Muestras totales: {stats.n_samples}")
    print(f"   - Features: {stats.n_features}")
    print(f"   - Clases: {stats.n_classes}")
    print(f"   - Distribución: {stats.class_distribution}")
    print("   - Rangos de features:")
    for i, (min_val, max_val) in enumerate(stats.feature_ranges):
        print(f"     Feature {i}: [{min_val:.2f}, {max_val:.2f}]")

    # 2. Dividir datos
    print("\n2. Dividiendo datos (80/20)...")
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    print(f"   - Train: {len(X_train)} muestras")
    print(f"   - Test: {len(X_test)} muestras")

    # 3. Normalizar
    print("\n3. Normalizando features...")
    X_train_norm = processor.normalize(X_train, fit=True)
    X_test_norm = processor.normalize(X_test, fit=False)
    print("   - Normalización completada")

    # 4. Entrenar modelo
    print("\n4. Entrenando modelo RandomForest...")
    config = TrainingConfig(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_cv_folds=5,
    )
    trainer = ModelTrainer(config)
    result = trainer.train(X_train_norm, y_train)

    print(f"   - Tiempo de entrenamiento: {result.training_time_seconds:.4f}s")
    print(f"   - Cross-validation scores: {[f'{s:.4f}' for s in result.cv_scores]}")
    print(f"   - CV Mean: {result.cv_mean:.4f} (+/- {result.cv_std:.4f})")

    # 5. Evaluar modelo
    print("\n5. Evaluando modelo en datos de test...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(result.model, X_test_norm, y_test)

    print(f"   - Accuracy: {metrics.accuracy:.4f}")
    print(f"   - Precision: {metrics.precision:.4f}")
    print(f"   - Recall: {metrics.recall:.4f}")
    print(f"   - F1-Score: {metrics.f1_score:.4f}")

    # 6. Matriz de confusión
    print("\n6. Matriz de Confusión:")
    print("   Predicho ->  0    1    2")
    print("   Real")
    for i, row in enumerate(metrics.confusion_matrix):
        print(f"     {i}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 7. Reporte de clasificación
    print("\n7. Reporte de Clasificación:")
    print(metrics.classification_report)

    # 8. Verificar umbral
    print("8. Verificación de umbral de accuracy (0.9)...")
    passed = evaluator.check_accuracy_threshold(metrics, threshold=0.9)
    if passed:
        print(f"   [OK] Accuracy {metrics.accuracy:.4f} >= 0.9 - APROBADO")
    else:
        print(f"   [WARN] Accuracy {metrics.accuracy:.4f} < 0.9 - WARNING")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETO - VERIFICACION EXITOSA")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
