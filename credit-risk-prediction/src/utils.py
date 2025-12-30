"""
Utility Functions for Credit Risk Prediction

This module contains helper functions used across the project.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')


def create_directories() -> None:
    """
    Create all necessary project directories if they don't exist.
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR,
        MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments to pass to pandas read functions
        
    Returns:
        DataFrame containing the loaded data
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    loaders = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet,
    }
    
    if file_extension not in loaders:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    logger.info(f"Loading data from {filepath}")
    return loaders[file_extension](filepath, **kwargs)


def save_data(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save DataFrame to various file formats.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
        **kwargs: Additional arguments to pass to pandas write functions
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    savers = {
        '.csv': df.to_csv,
        '.xlsx': df.to_excel,
        '.json': df.to_json,
        '.parquet': df.to_parquet,
    }
    
    if file_extension not in savers:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    logger.info(f"Saving data to {filepath}")
    if file_extension == '.csv':
        savers[file_extension](filepath, index=False, **kwargs)
    else:
        savers[file_extension](filepath, **kwargs)


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)


def generate_report(metrics: Dict[str, float], filepath: str) -> None:
    """
    Generate a JSON report with model metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        filepath: Path to save the report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Report saved to {filepath}")


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print summary information about a DataFrame.
    
    Args:
        df: DataFrame to summarize
        name: Name to display in the output
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values")
    print(f"{'='*60}\n")


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10
) -> tuple:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.
    
    Args:
        df: DataFrame containing the data
        feature: Name of the feature column
        target: Name of the target column
        bins: Number of bins for continuous features
        
    Returns:
        Tuple of (WoE DataFrame, IV value)
    """
    df_temp = df[[feature, target]].copy()
    
    # Bin continuous features
    if df_temp[feature].dtype in ['float64', 'int64']:
        df_temp['bin'] = pd.qcut(df_temp[feature], bins, duplicates='drop')
    else:
        df_temp['bin'] = df_temp[feature]
    
    # Calculate event and non-event counts
    grouped = df_temp.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']
    
    # Calculate distributions
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    
    grouped['event_dist'] = grouped['events'] / total_events
    grouped['non_event_dist'] = grouped['non_events'] / total_non_events
    
    # Add small epsilon to avoid division by zero
    eps = 0.0001
    grouped['event_dist'] = grouped['event_dist'].replace(0, eps)
    grouped['non_event_dist'] = grouped['non_event_dist'].replace(0, eps)
    
    # Calculate WoE and IV
    grouped['woe'] = np.log(grouped['non_event_dist'] / grouped['event_dist'])
    grouped['iv'] = (grouped['non_event_dist'] - grouped['event_dist']) * grouped['woe']
    
    iv = grouped['iv'].sum()
    
    return grouped, iv


def get_feature_importance_df(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    df_importance = df_importance.sort_values('importance', ascending=False)
    df_importance['importance_pct'] = df_importance['importance'] / df_importance['importance'].sum() * 100
    
    return df_importance.reset_index(drop=True)


if __name__ == "__main__":
    # Create project directories
    create_directories()
    print("Project directories created successfully!")
