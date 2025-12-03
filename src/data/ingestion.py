import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import great_expectations as ge
from loguru import logger
from ..config import settings

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a file path.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded pandas DataFrame
    """
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, expectations: Dict[str, Any]) -> Tuple[bool, Dict]:
    """
    Validate data using Great Expectations.
    
    Args:
        df: Input DataFrame
        expectations: Dictionary of expectations
        
    Returns:
        Tuple of (is_valid, validation_results)
    """
    try:
        # Create a Great Expectations dataset
        ge_df = ge.from_pandas(df)
        
        # Add expectations
        for col, exp in expectations.items():
            if exp['type'] == 'not_null':
                ge_df.expect_column_values_to_not_be_null(col)
            elif exp['type'] == 'in_range':
                ge_df.expect_column_values_to_be_between(
                    column=col,
                    min_value=exp.get('min'),
                    max_value=exp.get('max')
                )
            # Add more expectation types as needed
        
        # Validate
        validation = ge_df.validate()
        is_valid = validation['success']
        
        if not is_valid:
            logger.warning("Data validation failed")
            for result in validation['results']:
                if not result['success']:
                    logger.warning(f"Failed expectation: {result['expectation_config']['expectation_type']} "
                                 f"for column {result['expectation_config']['kwargs'].get('column')}")
        else:
            logger.info("Data validation passed")
            
        return is_valid, validation
        
    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        raise
