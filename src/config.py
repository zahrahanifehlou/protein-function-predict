from pydantic_settings import BaseSettings


from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Project structure
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    
    # Data paths
    RAW_DATA_PATH: Path = DATA_DIR / "raw"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"
    
    # Model parameters
    MODEL_NAME: str = "xgboost_model"
    TARGET_COLUMN: str = "go_term"
    SEQUENCE_FEATURES: List[str] = ["sequence_id"]
    K: int = 3
    
    # Training parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Feature engineering
    NUMERICAL_FEATURES: List[str] = [
        "feature1", "feature2", "feature3"  # Update with your actual feature names
    ]
    CATEGORICAL_FEATURES: List[str] = [
        "category1", "category2"  # Update with your actual categorical features
    ]
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "mlops-pipeline"
    
    # Monitoring
    DRIFT_THRESHOLD: float = 0.05
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings
settings = Settings()

# Create necessary directories
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
