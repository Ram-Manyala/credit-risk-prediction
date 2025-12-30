"""
Credit Risk Prediction Package

This package contains modules for building an end-to-end credit risk prediction system.

Modules:
    - data_ingestion: Load and validate data from various sources
    - data_cleaning: Preprocess and clean raw data
    - feature_engineering: Create and transform features
    - model_training: Train machine learning models
    - model_evaluation: Evaluate model performance
    - utils: Utility functions
"""

__version__ = '1.0.0'
__author__ = 'Credit Risk Team'

from . import data_ingestion
from . import data_cleaning
from . import feature_engineering
from . import model_training
from . import model_evaluation
from . import utils
