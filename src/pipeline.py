#!/usr/bin/env python3
"""
MLOps Pipeline: End-to-end machine learning pipeline for training and serving models.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import mlflow
from loguru import logger

# Import local modules
from .config import settings
from .data import load_data, validate_data, preprocess_data
from .models import train_model, evaluate_model, save_model, predict
from .monitoring import check_data_drift, monitor_performance, log_metrics




# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLOps Pipeline')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/raw/data.csv',
                        help='Path to input data file')
    parser.add_argument('--target-col', type=str, default=settings.TARGET_COLUMN,
                        help='Name of the target column')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='xgb',
                        choices=['xgb', 'lgbm'],
                        help='Type of model to train')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Type of machine learning task')
    
    # Training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # MLflow arguments
    parser.add_argument('--tracking-uri', type=str, default=settings.MLFLOW_TRACKING_URI,
                        help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default=settings.MLFLOW_EXPERIMENT_NAME,
                        help='MLflow experiment name')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    return parser.parse_args()

def main():
    """Main pipeline function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting MLOps Pipeline")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {mlflow.get_experiment_by_name(args.experiment_name).name}")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            'model_type': args.model_type,
            'task': args.task,
            'test_size': args.test_size,
            'random_state': args.random_state,
            'data_path': args.data_path,
            'target_column': args.target_col
        })
        
        # 1. Data Loading
        logger.info("\n" + "="*50)
        logger.info("1. Loading Data")
        logger.info("="*50)
        
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = load_data(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # 2. Data Validation
        logger.info("\n" + "="*50)
        logger.info("2. Validating Data")
        logger.info("="*50)
        
        # Example expectations (customize based on your data)
        expectations = {
            args.target_col: {'type': 'not_null'},
            # Add more expectations as needed
        }
        
     
        
        # 3. Data Preprocessing
        logger.info("\n" + "="*50)
        logger.info("3. Preprocessing Data")
        logger.info("="*50)
        
        X_train, X_test, y_train, y_test, preprocessor, mb = preprocess_data(
            df=df,
            target_column=args.target_col,
            sequence_features=settings.SEQUENCE_FEATURES,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save preprocessor
        preprocessor_path = output_dir / 'preprocessor.joblib'
        save_model(None, preprocessor, {}, output_dir, 'preprocessor')
        
        # 4. Model Training
        logger.info("\n" + "="*50)
        logger.info("4. Training Model")
        logger.info("="*50)
        
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=args.model_type,
            task=args.task
        )
        
        # 5. Model Evaluation
        logger.info("\n" + "="*50)
        logger.info("5. Evaluating Model")
        logger.info("="*50)
        
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            task=args.task
        )
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # 6. Save Model
        logger.info("\n" + "="*50)
        logger.info("6. Saving Model")
        logger.info("="*50)
        
        save_model(
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            model_dir=output_dir,
            model_name='model'
        )
        
        # 7. Model Monitoring (Example with test data)
        logger.info("\n" + "="*50)
        logger.info("7. Monitoring Model Performance")
        logger.info("="*50)
        
        # Make predictions on test set
        if args.task == 'classification':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        else:
            y_pred = model.predict(X_test)
            y_proba = None
        
        # Log performance metrics
        monitor_performance(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            task=args.task,
            model_name=f"{args.model_type}_{args.task}",
            output_dir=output_dir
        )
        
        # 8. Check for data drift (example with training and test data)
        logger.info("\n" + "="*50)
        logger.info("8. Checking for Data Drift")
        logger.info("="*50)
        
        # In a real scenario, you'd compare current data with a reference dataset
        drift_results = check_data_drift(
            reference_data=df,
            current_data=df.sample(frac=0.5, random_state=args.random_state),  # Example: sample half the data
            numerical_columns=settings.NUMERICAL_FEATURES,
            categorical_columns=settings.CATEGORICAL_FEATURES,
            threshold=settings.DRIFT_THRESHOLD
        )
        
        if drift_results['drift_detected']:
            logger.warning(f"Data drift detected in columns: {drift_results['drifted_columns']}")
            mlflow.log_metric('data_drift_detected', 1)
            mlflow.log_metric('num_drifted_columns', len(drift_results['drifted_columns']))
        else:
            logger.info("No significant data drift detected.")
            mlflow.log_metric('data_drift_detected', 0)
        
        logger.info("\n" + "="*50)
        logger.info("Pipeline completed successfully!")
        logger.info("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
