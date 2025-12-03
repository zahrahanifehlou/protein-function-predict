"""
Professional preprocessing pipeline for biological sequence data with multi-label GO terms.

Features:
- k-mer tokenization of protein/DNA sequences
- CountVectorizer + sparse scaling
- Multi-label binarization for GO terms
- Train/test split with reproducible preprocessing
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from loguru import logger
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning)


# ========================
# k-mer Transformation
# ========================

def sequence_to_kmer(sequence: str, k: int = 3) -> List[str]:
    """
    Convert a single sequence into overlapping k-mers.

    Args:
        sequence: Input biological sequence (e.g., protein or DNA string)
        k: Length of k-mer

    Returns:
        List of k-mers as strings

    Example:
        sequence_to_kmer("PEPTIDE", k=3) -> ['PEP', 'EPT', 'PTI', 'TID', 'IDE']
    """
    if not isinstance(sequence, str):
        raise ValueError(f"Expected string, got {type(sequence)}")
    if len(sequence) < k:
        logger.warning(f"Sequence shorter than k={k}: {sequence}")
        return []
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def sequence_to_kmer_list(sequences: List[str], k: int = 3) -> np.ndarray:
    """
    Apply k-mer transformation to a list of sequences and join into space-separated strings
    for CountVectorizer compatibility.

    Args:
        sequences: List/array of sequence strings
        k: k-mer size

    Returns:
        numpy array of space-joined k-mer strings
    """
    sequences = np.asarray(sequences).ravel()
    return np.array([" ".join(sequence_to_kmer(seq, k)) for seq in sequences])


class KmerTransformer(BaseEstimator, TransformerMixin):
    """
    Picklable transformer for k-mer conversion.
    """
    def __init__(self, k: int = 3):
        self.k = k
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return sequence_to_kmer_list(X, k=self.k)


# ========================
# Preprocessor Builders
# ========================

def create_sequence_preprocessor(sequence_features: List[str], k: int = 3) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for sequence features using k-mers.

    Args:
        sequence_features: List of column names containing sequences
        k: k-mer length

    Returns:
        ColumnTransformer with k-mer → CountVectorizer → StandardScaler (sparse-safe)
    """
    kmer_pipeline = Pipeline([
        ("kmer", KmerTransformer(k=k)),
        ("vectorizer", CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")),
        ("scaler", StandardScaler(with_mean=False))  # with_mean=False for sparse input
    ])

    return ColumnTransformer([
        ("seq", kmer_pipeline, sequence_features)
    ], remainder='drop', sparse_threshold=0.3)


def create_target_preprocessor() -> MultiLabelBinarizer:
    """
    Create a fitted MultiLabelBinarizer for GO terms (multi-label classification).

    Returns:
        Fitted MultiLabelBinarizer instance
    """
    return MultiLabelBinarizer(sparse_output=True)


# ========================
# Main Data Preparation
# ========================

def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    sequence_features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    k: int = 3
) -> Tuple[
    np.ndarray, np.ndarray,           # X_train, X_test
    np.ndarray, np.ndarray,           # y_train, y_test
    ColumnTransformer,                # preprocessor_X
    MultiLabelBinarizer               # mlb (fitted)
]:
    """
    Prepare sequence and multi-label GO term data for modeling.

    Args:
        df: Input DataFrame
        target_column: Name of column containing GO terms (as list or comma-separated string)
        sequence_features: List of column names with sequences
        test_size: Fraction of data for test split
        random_state: Random seed
        k: k-mer size

    Returns:
        X_train, X_test, y_train, y_test, preprocessor_X, mlb

    Raises:
        ValueError: If inputs are invalid
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in DataFrame")
    for col in sequence_features:
        if col not in df.columns:
            raise ValueError(f"Sequence column '{col}' not in DataFrame")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    logger.info(f"Preparing data: {len(df)} samples, k={k}, test_size={test_size}")

    # Extract features and target
    df=df.sample(2000)
    X = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # Handle GO terms: convert to list of lists
    if isinstance(y_raw.iloc[0], str):
        y = y_raw.str.split(r'[,;\s]+').apply(lambda x: [term.strip() for term in x if term.strip()])
    elif isinstance(y_raw.iloc[0], (list, np.ndarray)):
        y = y_raw
    else:
        raise ValueError("GO terms must be comma-separated strings or lists")

    # Split data first
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    # Fit preprocessors
    preprocessor_X = create_sequence_preprocessor(sequence_features, k=k)
    mlb = create_target_preprocessor()

    # Transform sequences
    X_train_transformed = preprocessor_X.fit_transform(X_train)
    X_test_transformed = preprocessor_X.transform(X_test)

    # Transform labels
    y_train = mlb.fit_transform(y_train_raw)
    y_test = mlb.transform(y_test_raw)

    logger.info(f"X_train shape: {X_train_transformed.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test_transformed.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Number of GO classes: {len(mlb.classes_)}")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_X, mlb




def save_preprocessor(preprocessor: ColumnTransformer, file_path: Path) -> None:
    """Save the preprocessor to disk."""
    joblib.dump(preprocessor, file_path)
    logger.info(f"Saved preprocessor to {file_path}")

def load_preprocessor(file_path: Path) -> ColumnTransformer:
    """Load a preprocessor from disk."""
    return joblib.load(file_path)
