"""Property tests para entrenamiento de modelos.

**Validates: Requirements 4.2, 4.3**
"""

import numpy as np
from hypothesis import given, settings, strategies as st
from sklearn.datasets import make_classification

from src.ml_lambda.training.evaluator import ModelEvaluator
from src.ml_lambda.training.trainer import ModelTrainer, TrainingConfig


class TestTrainingMetricsProperties:
    """Property tests para métricas de entrenamiento.

    **Property 3: Entrenamiento Produce Métricas Válidas**

    Para cualquier modelo entrenado y evaluado:
    - Accuracy, precision, recall y f1-score deben estar en el rango [0, 1]
    - La matriz de confusión debe tener dimensiones (n_classes, n_classes)
    - La suma de todos los elementos de la matriz de confusión debe ser igual
      al número de muestras de test

    **Validates: Requirements 4.2, 4.3**
    """

    @settings(max_examples=25, deadline=None)
    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        n_classes=st.integers(min_value=2, max_value=4),
        n_estimators=st.integers(min_value=5, max_value=15),
    )
    def test_metrics_in_valid_range(
        self, n_samples: int, n_classes: int, n_estimators: int
    ):
        """
        Property 3: Para cualquier modelo entrenado, las métricas deben estar
        en el rango [0, 1].

        **Validates: Requirements 4.2, 4.3**
        """
        # Generar dataset sintético
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Entrenar modelo
        config = TrainingConfig(n_estimators=n_estimators, random_state=42)
        trainer = ModelTrainer(config)
        trainer.train(X, y)

        # Evaluar
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X, y)

        # Verificar rangos
        assert 0.0 <= metrics.accuracy <= 1.0, f"accuracy={metrics.accuracy}"
        assert 0.0 <= metrics.precision <= 1.0, f"precision={metrics.precision}"
        assert 0.0 <= metrics.recall <= 1.0, f"recall={metrics.recall}"
        assert 0.0 <= metrics.f1_score <= 1.0, f"f1_score={metrics.f1_score}"

    @settings(max_examples=25, deadline=None)
    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        n_classes=st.integers(min_value=2, max_value=4),
    )
    def test_confusion_matrix_dimensions(self, n_samples: int, n_classes: int):
        """
        Property 3: La matriz de confusión debe tener dimensiones (n_classes, n_classes).

        **Validates: Requirements 4.2, 4.3**
        """
        # Generar dataset sintético
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Entrenar modelo
        config = TrainingConfig(n_estimators=10, random_state=42)
        trainer = ModelTrainer(config)
        trainer.train(X, y)

        # Evaluar
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X, y)

        # Verificar dimensiones
        assert metrics.confusion_matrix.shape == (n_classes, n_classes)

    @settings(max_examples=25, deadline=None)
    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        n_classes=st.integers(min_value=2, max_value=4),
    )
    def test_confusion_matrix_sum_equals_samples(self, n_samples: int, n_classes: int):
        """
        Property 3: La suma de la matriz de confusión debe ser igual al número
        de muestras de test.

        **Validates: Requirements 4.2, 4.3**
        """
        # Generar dataset sintético
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Entrenar modelo
        config = TrainingConfig(n_estimators=10, random_state=42)
        trainer = ModelTrainer(config)
        trainer.train(X, y)

        # Evaluar
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer.model, X, y)

        # Verificar suma
        assert metrics.confusion_matrix.sum() == n_samples
