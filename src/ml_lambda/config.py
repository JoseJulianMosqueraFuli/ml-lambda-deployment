"""Configuración centralizada del proyecto."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuración centralizada del proyecto."""

    # Versión
    version: str = "v1.0.0"

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    model_filename: str = "model.joblib"
    metadata_filename: str = "model_metadata.json"

    # Data
    test_size: float = 0.2
    random_state: int = 42

    # Training
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    n_cv_folds: int = 5
    accuracy_threshold: float = 0.9

    # Inference
    expected_features: int = 4
    max_body_size: int = 1024  # 1KB

    # AWS
    aws_region: str = "us-east-1"
    lambda_timeout: int = 30
    lambda_memory: int = 256

    # Logging
    log_level: str = "INFO"

    # Feature names (Iris dataset)
    feature_names: list[str] = field(
        default_factory=lambda: [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )

    # Class names (Iris dataset)
    class_names: list[str] = field(
        default_factory=lambda: ["setosa", "versicolor", "virginica"]
    )

    @property
    def model_path(self) -> Path:
        """Ruta completa al modelo serializado."""
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        """Ruta completa a los metadatos del modelo."""
        return self.artifacts_dir / self.metadata_filename


# Instancia global de configuración
config = Config()
