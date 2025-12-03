import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
from loguru import logger

from ..config import settings

def load_model(
    model_path: Path,
    preprocessor_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load a trained model and its preprocessor from disk.
    
    Args:
        model_path: Path to the saved model
        preprocessor_path: Optional path to the saved preprocessor
        
    Returns:
        Dictionary containing 'model' and 'preprocessor' (if available)
    """
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path else None
        
        logger.info(f"Loaded model from {model_path}")
        if preprocessor_path:
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            
        return {
            'model': model,
            'preprocessor': preprocessor
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray, List[Dict]],
    preprocessor: Optional[Any] = None,
    return_proba: bool = False,
    threshold: float = 0.5
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        data: Input data for prediction
        preprocessor: Optional preprocessor to transform the input data
        return_proba: Whether to return class probabilities (for classification)
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Model predictions
    """
    try:
        # Convert input to DataFrame if it's a list of dicts
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Apply preprocessing if preprocessor is provided
        if preprocessor is not None:
            data = preprocessor.transform(data)
            
            # If the output is a sparse matrix, convert to dense
            if hasattr(data, 'toarray'):
                data = data.toarray()
        
        # Make predictions
        if return_proba and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(data)
            predictions = (proba[:, 1] >= threshold).astype(int)
            return {
                'predictions': predictions,
                'probabilities': proba
            }
        else:
            predictions = model.predict(data)
            
            if hasattr(model, 'predict_proba') and return_proba:
                proba = model.predict_proba(data)
                return {
                    'predictions': predictions,
                    'probabilities': proba
                }
            
            return predictions
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def batch_predict(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    preprocessor_path: Optional[Path] = None,
    batch_size: int = 1000,
    **predict_kwargs
) -> None:
    """
    Make predictions in batches for large datasets.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to the input data file
        output_path: Path to save the predictions
        preprocessor_path: Optional path to the saved preprocessor
        batch_size: Number of samples per batch
        **predict_kwargs: Additional arguments to pass to the predict function
    """
    try:
        # Load model and preprocessor
        artifacts = load_model(model_path, preprocessor_path)
        model = artifacts['model']
        preprocessor = artifacts.get('preprocessor')
        
        # Initialize output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_batch = True
        
        # Process data in batches
        for chunk in pd.read_csv(data_path, chunksize=batch_size):
            # Make predictions
            predictions = predict(
                model=model,
                data=chunk,
                preprocessor=preprocessor,
                **predict_kwargs
            )
            
            # Convert predictions to DataFrame
            if isinstance(predictions, dict):
                result_df = pd.DataFrame(predictions)
            else:
                result_df = pd.DataFrame({'predictions': predictions})
            
            # Save results (append if not first batch)
            result_df.to_csv(
                output_path,
                mode='a',
                header=first_batch,
                index=False
            )
            first_batch = False
            
        logger.info(f"Saved predictions to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise
