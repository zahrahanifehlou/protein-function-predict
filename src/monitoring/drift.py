import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from scipy import stats
from loguru import logger

from ..config import settings

def check_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list,
    threshold: float = 0.05,
    method: str = 'ks'
) -> Dict[str, Any]:
    """
    Check for data drift between reference and current data.
    
    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset to compare against reference
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        threshold: Significance threshold for drift detection
        method: Drift detection method ('ks' for Kolmogorov-Smirnov or 'cvm' for Cramér-von Mises)
        
    Returns:
        Dictionary containing drift detection results
    """
    try:
        results = {
            'drift_detected': False,
            'drifted_columns': [],
            'p_values': {},
            'test_statistics': {}
        }
        
        # Check numerical columns
        for col in numerical_columns:
            if col not in reference_data.columns or col not in current_data.columns:
                logger.warning(f"Column {col} not found in both datasets")
                continue
                
            ref_data = reference_data[col].dropna().values
            curr_data = current_data[col].dropna().values
            
            if method == 'ks':
                stat, p_value = stats.ks_2samp(ref_data, curr_data)
            elif method == 'cvm':
                stat, p_value = stats.cramervonmises_2samp(ref_data, curr_data)
            else:
                # Default to KS test if method not recognized
                logger.warning(f"Method '{method}' not supported, using 'ks' instead")
                stat, p_value = stats.ks_2samp(ref_data, curr_data)
            
            results['p_values'][col] = p_value
            results['test_statistics'][col] = stat
            
            if p_value < threshold:
                results['drift_detected'] = True
                results['drifted_columns'].append(col)
                logger.warning(f"Drift detected in column {col} (p-value: {p_value:.4f})")
        
        # Check categorical columns
        for col in categorical_columns:
            if col not in reference_data.columns or col not in current_data.columns:
                logger.warning(f"Column {col} not found in both datasets")
                continue
                
            # Convert to categorical codes for chi2 test
            all_categories = list(set(reference_data[col].unique()) | set(current_data[col].unique()))
            ref_codes = reference_data[col].astype('category', categories=all_categories).cat.codes
            curr_codes = current_data[col].astype('category', categories=all_categories).cat.codes
            
            # Create contingency table
            ref_counts = ref_codes.value_counts().sort_index()
            curr_counts = curr_codes.value_counts().sort_index()
            
            # Ensure all categories are represented in both datasets
            all_indices = sorted(set(ref_counts.index) | set(curr_counts.index))
            ref_counts = ref_counts.reindex(all_indices, fill_value=0)
            curr_counts = curr_counts.reindex(all_indices, fill_value=0)
            
            # Perform chi-square test
            chi2, p_value, dof, _ = stats.chi2_contingency([ref_counts, curr_counts])
            
            results['p_values'][col] = p_value
            results['test_statistics'][col] = chi2
            
            if p_value < threshold:
                results['drift_detected'] = True
                results['drifted_columns'].append(col)
                logger.warning(f"Drift detected in categorical column {col} (p-value: {p_value:.4f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in data drift detection: {str(e)}")
        raise

def check_concept_drift(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    reference_metric: float,
    metric_fn: callable,
    threshold: float = 0.1,
    task: str = 'classification'
) -> Dict[str, Any]:
    """
    Check for concept drift by monitoring model performance.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        reference_metric: Reference performance metric value
        metric_fn: Function to calculate the performance metric
        threshold: Relative threshold for performance degradation
        task: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary containing concept drift detection results
    """
    try:
        # Make predictions
        if task == 'classification':
            y_pred = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
        else:  # regression
            y_pred = model.predict(X)
        
        # Calculate current metric
        current_metric = metric_fn(y, y_pred)
        
        # Check for significant degradation
        relative_change = abs(current_metric - reference_metric) / reference_metric
        drift_detected = relative_change > threshold
        
        results = {
            'drift_detected': drift_detected,
            'reference_metric': reference_metric,
            'current_metric': current_metric,
            'relative_change': relative_change,
            'threshold': threshold
        }
        
        if drift_detected:
            logger.warning(
                f"Concept drift detected! Performance changed by {relative_change*100:.2f}% "
                f"(reference: {reference_metric:.4f}, current: {current_metric:.4f})"
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in concept drift detection: {str(e)}")
        raise
