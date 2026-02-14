#!/usr/bin/env python3
"""Script to verify model serialization and prediction consistency."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_lambda.data.processor import DataProcessor
from ml_lambda.training.trainer import ModelTrainer, TrainingConfig
from ml_lambda.training.evaluator import ModelEvaluator
from ml_lambda.model.serializer import ModelSerializer, ModelMetadata
from ml_lambda.utils.logging import StructuredLogger

logger = StructuredLogger("verify_serialization")


def main():
    """Verify serialization round-trip preserves predictions."""
    logger.info("Starting serialization verification")
    
    # 1. Load and prepare data
    logger.info("Loading dataset")
    processor = DataProcessor()
    X, y = processor.load_iris()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train = processor.normalize(X_train, fit=True)
    X_test = processor.normalize(X_test, fit=False)
    
    # 2. Train model
    logger.info("Training model")
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    result = trainer.train(X_train, y_train)
    
    # 3. Evaluate model
    logger.info("Evaluating model")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(result.model, X_test, y_test)
    
    logger.info(
        "Model trained",
        accuracy=metrics.accuracy,
        cv_mean=result.cv_mean
    )
    
    # 4. Make predictions with original model
    logger.info("Making predictions with original model")
    original_predictions = result.model.predict(X_test)
    original_probabilities = result.model.predict_proba(X_test)
    
    # 5. Save model
    logger.info("Saving model")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    
    from datetime import datetime
    
    metadata = ModelMetadata(
        version="v1.0.0",
        created_at=datetime.utcnow(),
        accuracy=metrics.accuracy,
        n_features=4,
        n_classes=3,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        class_names=["setosa", "versicolor", "virginica"],
        training_config={
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
            "random_state": config.random_state
        }
    )
    
    serializer = ModelSerializer()
    model_hash = serializer.save(result.model, metadata, model_path)
    
    logger.info("Model saved", path=str(model_path), hash=model_hash)
    
    # 6. Load model
    logger.info("Loading model")
    loaded = serializer.load(model_path)
    
    logger.info(
        "Model loaded",
        version=loaded.metadata.version,
        accuracy=loaded.metadata.accuracy
    )
    
    # 7. Make predictions with loaded model
    logger.info("Making predictions with loaded model")
    loaded_predictions = loaded.model.predict(X_test)
    loaded_probabilities = loaded.model.predict_proba(X_test)
    
    # 8. Verify predictions are identical
    logger.info("Verifying predictions")
    
    predictions_match = np.array_equal(original_predictions, loaded_predictions)
    probabilities_match = np.allclose(original_probabilities, loaded_probabilities)
    
    if predictions_match and probabilities_match:
        logger.info("✓ Verification PASSED: Predictions are identical after serialization")
        print("\n" + "="*60)
        print("✓ SERIALIZATION VERIFICATION PASSED")
        print("="*60)
        print(f"\nModel saved to: {model_path}")
        print(f"Model hash: {model_hash}")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"CV Score: {result.cv_mean:.4f} ± {result.cv_std:.4f}")
        print(f"\nTest samples: {len(X_test)}")
        print(f"Predictions match: {predictions_match}")
        print(f"Probabilities match: {probabilities_match}")
        print("\nSample predictions (first 5):")
        for i in range(min(5, len(X_test))):
            print(f"  Sample {i+1}: {original_predictions[i]} (prob: {original_probabilities[i].max():.3f})")
        print("\n" + "="*60)
        return 0
    else:
        logger.error(
            "✗ Verification FAILED: Predictions differ after serialization",
            predictions_match=predictions_match,
            probabilities_match=probabilities_match
        )
        print("\n" + "="*60)
        print("✗ SERIALIZATION VERIFICATION FAILED")
        print("="*60)
        print(f"Predictions match: {predictions_match}")
        print(f"Probabilities match: {probabilities_match}")
        
        if not predictions_match:
            diff_count = np.sum(original_predictions != loaded_predictions)
            print(f"\nDifferent predictions: {diff_count}/{len(X_test)}")
        
        if not probabilities_match:
            max_diff = np.max(np.abs(original_probabilities - loaded_probabilities))
            print(f"Max probability difference: {max_diff}")
        
        print("\n" + "="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
