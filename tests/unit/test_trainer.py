"""Tests unitarios para ModelTrainer y ModelEvaluator."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris

from src.ml_lambda.training.evaluator import EvaluationMetrics, ModelEvaluator
from src.ml_lambda.training.trainer import ModelTrainer, TrainingConfig, TrainingResult
from src.ml_lambda.utils.exceptions import ModelNotTrainedError


class TestModelTrainer:
    """Tests para ModelTrainer."""

    @pytest.fixture
    def iris_data(self):
        """Fixture con datos de Iris para entrenamiento."""
        data = load_iris()
        return data.data, data.target

    @pytest.fixture
    def default_config(self):
        """Configuración por defecto."""
        return TrainingConfig()

    def test_train_returns_training_result(self, iris_data, default_config):
        """Verifica que train() retorna TrainingResult."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)

        result = trainer.train(X, y)

        assert isinstance(result, TrainingResult)

    def test_train_model_is_fitted(self, iris_data, default_config):
        """Verifica que el modelo queda entrenado."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)

        result = trainer.train(X, y)

        # El modelo debe poder hacer predicciones
        predictions = result.model.predict(X[:5])
        assert len(predictions) == 5

    def test_train_cv_scores_correct_length(self, iris_data, default_config):
        """Verifica que cv_scores tiene la longitud correcta."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)

        result = trainer.train(X, y)

        assert len(result.cv_scores) == default_config.n_cv_folds

    def test_train_cv_mean_in_valid_range(self, iris_data, default_config):
        """Verifica que cv_mean está en rango válido."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)

        result = trainer.train(X, y)

        assert 0.0 <= result.cv_mean <= 1.0

    def test_train_records_training_time(self, iris_data, default_config):
        """Verifica que se registra el tiempo de entrenamiento."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)

        result = trainer.train(X, y)

        assert result.training_time_seconds > 0

    def test_model_property_raises_before_training(self, default_config):
        """Verifica que model property lanza error si no está entrenado."""
        trainer = ModelTrainer(default_config)

        with pytest.raises(ModelNotTrainedError):
            _ = trainer.model

    def test_model_property_returns_model_after_training(self, iris_data, default_config):
        """Verifica que model property retorna modelo después de entrenar."""
        X, y = iris_data
        trainer = ModelTrainer(default_config)
        trainer.train(X, y)

        model = trainer.model

        assert model is not None

    def test_custom_config_is_applied(self, iris_data):
        """Verifica que la configuración personalizada se aplica."""
        X, y = iris_data
        config = TrainingConfig(n_estimators=10, max_depth=3, n_cv_folds=3)
        trainer = ModelTrainer(config)

        result = trainer.train(X, y)

        assert result.model.n_estimators == 10
        assert result.model.max_depth == 3
        assert len(result.cv_scores) == 3


class TestModelEvaluator:
    """Tests para ModelEvaluator."""

    @pytest.fixture
    def trained_model(self):
        """Fixture con modelo entrenado."""
        data = load_iris()
        X, y = data.data, data.target
        trainer = ModelTrainer(TrainingConfig(n_estimators=10))
        result = trainer.train(X, y)
        return result.model, X, y

    def test_evaluate_returns_metrics(self, trained_model):
        """Verifica que evaluate() retorna EvaluationMetrics."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert isinstance(metrics, EvaluationMetrics)

    def test_evaluate_accuracy_in_valid_range(self, trained_model):
        """Verifica que accuracy está en rango [0, 1]."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert 0.0 <= metrics.accuracy <= 1.0

    def test_evaluate_precision_in_valid_range(self, trained_model):
        """Verifica que precision está en rango [0, 1]."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert 0.0 <= metrics.precision <= 1.0

    def test_evaluate_recall_in_valid_range(self, trained_model):
        """Verifica que recall está en rango [0, 1]."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert 0.0 <= metrics.recall <= 1.0

    def test_evaluate_f1_in_valid_range(self, trained_model):
        """Verifica que f1_score está en rango [0, 1]."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert 0.0 <= metrics.f1_score <= 1.0

    def test_evaluate_confusion_matrix_shape(self, trained_model):
        """Verifica que la matriz de confusión tiene forma correcta."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        n_classes = len(np.unique(y))

        metrics = evaluator.evaluate(model, X, y)

        assert metrics.confusion_matrix.shape == (n_classes, n_classes)

    def test_evaluate_confusion_matrix_sum(self, trained_model):
        """Verifica que la suma de la matriz de confusión es n_samples."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert metrics.confusion_matrix.sum() == len(y)

    def test_evaluate_classification_report_not_empty(self, trained_model):
        """Verifica que classification_report no está vacío."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X, y)

        assert len(metrics.classification_report) > 0

    def test_check_accuracy_threshold_passes(self, trained_model):
        """Verifica que check_accuracy_threshold pasa con accuracy alto."""
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X, y)

        # Con datos de entrenamiento, accuracy debería ser alto
        result = evaluator.check_accuracy_threshold(metrics, threshold=0.5)

        assert result is True

    def test_check_accuracy_threshold_warns_on_low_accuracy(self):
        """Verifica que se emite warning cuando accuracy es bajo."""
        evaluator = ModelEvaluator()
        # Crear métricas con accuracy bajo
        metrics = EvaluationMetrics(
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            confusion_matrix=np.array([[25, 25], [25, 25]]),
            classification_report="",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = evaluator.check_accuracy_threshold(metrics, threshold=0.9)

            assert result is False
            assert len(w) == 1
            assert "por debajo del umbral" in str(w[0].message)
