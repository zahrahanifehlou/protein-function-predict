import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, make_scorer,
    hamming_loss, jaccard_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
from scipy.sparse import issparse

from ..config import settings

def get_model(model_type: str = 'xgb', task: str = 'classification', **kwargs) -> Any:
    """
    Initialize a machine learning model.
    
    Args:
        model_type: Type of model ('xgb' or 'lgbm')
        task: Type of task ('classification' or 'regression')
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Initialized model
    """
    model_params = {
        'random_state': settings.RANDOM_STATE,
        'n_jobs': -1,
        **kwargs
    }
    
    if model_type == 'xgb':
        if task == 'classification':
            return XGBClassifier(**model_params)
        else:
            return XGBRegressor(**model_params)
    elif model_type == 'lgbm':
        if task == 'classification':
            return LGBMClassifier(**model_params)
        else:
            return LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'xgb',
    task: str = 'classification',
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Any:
    """
    Train a machine learning model with hyperparameter tuning.
    Supports multi-label classification.
    
    Args:
        X_train: Training features
        y_train: Training target (can be multi-label sparse matrix)
        model_type: Type of model ('xgb' or 'lgbm')
        task: Type of task ('classification' or 'regression')
        param_grid: Hyperparameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Trained model with best parameters
    """
    try:
        # Check if multi-label classification
        is_multilabel = issparse(y_train) or (isinstance(y_train, np.ndarray) and y_train.ndim > 1 and y_train.shape[1] > 1)
        
        if is_multilabel:
            logger.info("Detected multi-label classification task")
            # For multi-label, use simpler training without GridSearchCV
            # GridSearchCV doesn't work well with multi-label sparse matrices
            
            # Convert sparse y_train to dense array for XGBoost compatibility
            y_train_dense = y_train.toarray() if issparse(y_train) else y_train
            logger.info(f"Converted y_train to dense array: shape {y_train_dense.shape}")
            
            if model_type == 'xgb':
                base_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=1,  # Set to 1 to avoid parallel issues with MultiOutputClassifier
                    tree_method='hist'
                )
            else:  # lgbm
                base_model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=1  # Set to 1 to avoid parallel issues with MultiOutputClassifier
                )
            
            # Wrap with MultiOutputClassifier for multi-label
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            
            logger.info(f"Training {model_type.upper()} multi-label model...")
            model.fit(X_train, y_train_dense)
            logger.info("Multi-label model training completed")
            
            return model
        else:
            # Single-label classification or regression with GridSearchCV
            if param_grid is None:
                if model_type == 'xgb':
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                else:  # lgbm
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'num_leaves': [31, 63, 127]
                    }
            
            # Initialize model
            model = get_model(model_type=model_type, task=task)
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # Train model with hyperparameter tuning
            logger.info(f"Training {model_type.upper()} model with {cv}-fold cross-validation...")
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'classification'
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    Supports multi-label classification.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target (can be multi-label sparse matrix)
        task: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        # Check if multi-label
        is_multilabel = issparse(y_test) or (isinstance(y_test, np.ndarray) and y_test.ndim > 1 and y_test.shape[1] > 1)
        
        if task == 'classification':
            if is_multilabel:
                # Multi-label metrics
                # Convert sparse to dense if needed for metrics
                y_test_dense = y_test.toarray() if issparse(y_test) else y_test
                y_pred_dense = y_pred.toarray() if issparse(y_pred) else y_pred
                
                metrics.update({
                    'hamming_loss': hamming_loss(y_test_dense, y_pred_dense),
                    'jaccard_score': jaccard_score(y_test_dense, y_pred_dense, average='samples', zero_division=0),
                    'precision': precision_score(y_test_dense, y_pred_dense, average='samples', zero_division=0),
                    'recall': recall_score(y_test_dense, y_pred_dense, average='samples', zero_division=0),
                    'f1': f1_score(y_test_dense, y_pred_dense, average='samples', zero_division=0)
                })
            else:
                # Single-label metrics
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                metrics.update({
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                })
                
                if y_proba is not None and len(np.unique(y_test)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                    
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            })
        
        # Log metrics
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def save_model(
    model: Any,
    preprocessor: Any,
    metrics: Dict[str, float],
    model_dir: Path,
    model_name: str = 'model'
) -> None:
    """
    Save a trained model and its preprocessor to disk.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        metrics: Dictionary of evaluation metrics
        model_dir: Directory to save the model
        model_name: Base name for the model files
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Save preprocessor
        preprocessor_path = model_dir / f"{model_name}_preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        
        # Save metrics
        metrics_path = model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved model and artifacts to {model_dir}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
