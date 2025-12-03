import time
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from loguru import logger

from ..config import settings

def log_metrics(
    metrics: Dict[str, float],
    model_name: str,
    model_version: str = "1.0",
    stage: str = "Production",
    tags: Optional[Dict[str, str]] = None,
    experiment_name: Optional[str] = None
) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        model_name: Name of the model
        model_version: Version of the model
        stage: Model stage (e.g., 'Staging', 'Production')
        tags: Additional tags to log
        experiment_name: Optional MLflow experiment name
    """
    try:
        # Set experiment if specified
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        # Check if there's already an active run
        active_run = mlflow.active_run()
        
        if active_run:
            # Use existing run
            # Log parameters
            mlflow.log_params({
                'model_name': model_name,
                'model_version': model_version,
                'stage': stage
            })
            
            # Log metrics (filter out non-numeric values)
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
            mlflow.log_metrics(numeric_metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            logger.info(f"Logged metrics to existing MLflow run {active_run.info.run_id}")
        else:
            # Start new MLflow run
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params({
                    'model_name': model_name,
                    'model_version': model_version,
                    'stage': stage
                })
                
                # Log metrics (filter out non-numeric values)
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
                mlflow.log_metrics(numeric_metrics)
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                logger.info(f"Logged metrics to MLflow run {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"Error logging metrics to MLflow: {str(e)}")
        raise

def monitor_performance(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    task: str = 'classification',
    model_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Calculate and log performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for classification)
        task: Type of task ('classification' or 'regression')
        model_name: Optional model name for logging
        output_dir: Optional directory to save metrics
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_squared_error, r2_score, classification_report,
            confusion_matrix, hamming_loss, jaccard_score
        )
        from scipy.sparse import issparse
        
        metrics = {}
        
        # Check if multi-label
        is_multilabel = issparse(y_true) or (isinstance(y_true, np.ndarray) and y_true.ndim > 1 and y_true.shape[1] > 1)
        
        if task == 'classification':
            if is_multilabel:
                # Multi-label classification metrics
                # Convert sparse to dense if needed
                y_true_dense = y_true.toarray() if issparse(y_true) else y_true
                y_pred_dense = y_pred.toarray() if issparse(y_pred) else y_pred
                
                metrics.update({
                    'hamming_loss': hamming_loss(y_true_dense, y_pred_dense),
                    'jaccard_score': jaccard_score(y_true_dense, y_pred_dense, average='samples', zero_division=0),
                    'precision': precision_score(y_true_dense, y_pred_dense, average='samples', zero_division=0),
                    'recall': recall_score(y_true_dense, y_pred_dense, average='samples', zero_division=0),
                    'f1': f1_score(y_true_dense, y_pred_dense, average='samples', zero_division=0)
                })
                
                logger.info("Multi-label classification metrics calculated")
            else:
                # Single-label classification metrics
                # Basic metrics
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                })
                
                # ROC-AUC if probabilities are available
                if y_proba is not None and len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                
                # Additional classification report metrics
                report = classification_report(
                    y_true, y_pred, output_dict=True, zero_division=0
                )
                
                # Log per-class metrics if not too many classes
                if len(report) < 10:  # Arbitrary threshold to avoid cluttering
                    for label, scores in report.items():
                        if label.isdigit() or label in ['micro avg', 'macro avg', 'weighted avg']:
                            for metric, value in scores.items():
                                if metric != 'support':
                                    metrics[f"{label}_{metric}"] = value
                
                # Confusion matrix (only for single-label)
                cm = confusion_matrix(y_true, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
            
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': np.mean(np.abs(y_true - y_pred)),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
            })
        
        # Log metrics to console
        logger.info("\n" + "="*50)
        logger.info("Performance Metrics:")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{name}: {value:.4f}")
        logger.info("="*50 + "\n")
        
        # Save metrics to file if output directory is provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = output_dir / f"metrics_{timestamp}.json"
            
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                metrics_serializable = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics_serializable[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        metrics_serializable[k] = v.tolist()
                    else:
                        metrics_serializable[k] = v
                
                json.dump(metrics_serializable, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Log to MLflow if model_name is provided
        if model_name:
            log_metrics(
                metrics=metrics,
                model_name=model_name,
                tags={
                    'task': task,
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in performance monitoring: {str(e)}")
        raise

def log_feature_importance(
    model: Any,
    feature_names: list,
    importance_type: str = 'gain',
    top_n: int = 20
) -> Dict[str, float]:
    """
    Extract and log feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or get_booster() attribute
        feature_names: List of feature names
        importance_type: Type of importance to extract ('gain', 'weight', 'cover' for tree-based models)
        top_n: Number of top features to return
        
    Returns:
        Dictionary of feature importances
    """
    try:
        importances = {}
        
        # XGBoost
        if hasattr(model, 'get_booster'):
            importance_scores = model.get_booster().get_score(importance_type=importance_type)
            # Convert to dictionary with feature names as keys
            importances = {feature_names[int(k[1:])]: v for k, v in importance_scores.items()}
        # scikit-learn
        elif hasattr(model, 'feature_importances_'):
            importances = dict(zip(feature_names, model.feature_importances_))
        # LightGBM
        elif hasattr(model, 'feature_importance'):
            importances = dict(zip(feature_names, model.feature_importance(importance_type=importance_type)))
        else:
            logger.warning("Model does not support feature importance calculation")
            return {}
        
        # Sort by importance
        sorted_importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        # Log to console
        logger.info("\n" + "="*50)
        logger.info(f"Top {top_n} Feature Importances:")
        for feature, importance in sorted_importances.items():
            logger.info(f"{feature}: {importance:.6f}")
        logger.info("="*50 + "\n")
        
        # Log to MLflow
        if mlflow.active_run():
            for feature, importance in sorted_importances.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        return sorted_importances
        
    except Exception as e:
        logger.error(f"Error logging feature importance: {str(e)}")
        return {}
