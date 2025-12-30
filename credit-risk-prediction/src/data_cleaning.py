"""
Data Cleaning Module for Credit Risk Prediction

This module handles:
- Missing value treatment
- Outlier detection and handling
- Data type corrections
- Data quality improvements
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from .utils import (
    load_data, save_data, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    print_dataframe_info, logger, create_directories
)


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for credit risk data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data cleaner.
        
        Args:
            df: Raw DataFrame to clean
        """
        self.df = df.copy()
        self.cleaning_report = {
            'initial_rows': len(df),
            'initial_cols': len(df.columns),
            'missing_handled': {},
            'outliers_handled': {},
            'duplicates_removed': 0,
            'final_rows': 0,
            'final_cols': 0
        }
        
    def handle_missing_values(self, strategy: str = 'smart') -> 'DataCleaner':
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values
                     - 'smart': Use appropriate strategy per column type
                     - 'drop': Drop rows with any missing values
                     - 'fill_median': Fill numeric with median
                     - 'fill_mode': Fill categorical with mode
                     
        Returns:
            Self for method chaining
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'smart':
            self._smart_imputation()
        elif strategy == 'fill_median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].median()
            )
        elif strategy == 'fill_mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_handled'] = {
            'before': missing_before,
            'after': missing_after,
            'strategy': strategy
        }
        
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        return self
    
    def _smart_imputation(self) -> None:
        """
        Apply smart imputation based on column type and distribution.
        """
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
                
            if self.df[col].dtype in ['float64', 'int64']:
                # Check skewness for numeric columns
                skewness = self.df[col].skew()
                if abs(skewness) > 1:
                    # Use median for skewed data
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    # Use mean for normally distributed data
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
            else:
                # Use mode for categorical
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
    
    def handle_outliers(
        self, 
        method: str = 'iqr',
        columns: Optional[List[str]] = None,
        threshold: float = 1.5
    ) -> 'DataCleaner':
        """
        Handle outliers in numeric columns.
        
        Args:
            method: Method for outlier detection
                   - 'iqr': Interquartile range method
                   - 'zscore': Z-score method
                   - 'percentile': Percentile capping
            columns: Specific columns to process (None = all numeric)
            threshold: Threshold for outlier detection
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Handling outliers with method: {method}")
        
        if columns is None:
            # Get numeric columns, excluding ID and target
            exclude_cols = ['customer_id', 'default', 'application_date']
            columns = [
                col for col in self.df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        
        outliers_handled = {}
        
        for col in columns:
            outliers_before = self._count_outliers(col, method, threshold)
            
            if method == 'iqr':
                self._handle_outliers_iqr(col, threshold)
            elif method == 'zscore':
                self._handle_outliers_zscore(col, threshold)
            elif method == 'percentile':
                self._handle_outliers_percentile(col)
            
            outliers_after = self._count_outliers(col, method, threshold)
            
            if outliers_before > 0:
                outliers_handled[col] = {
                    'before': outliers_before,
                    'after': outliers_after
                }
        
        self.cleaning_report['outliers_handled'] = outliers_handled
        logger.info(f"Outliers handled in {len(outliers_handled)} columns")
        return self
    
    def _count_outliers(self, col: str, method: str, threshold: float) -> int:
        """
        Count outliers in a column.
        """
        if method == 'iqr':
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        elif method == 'zscore':
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            return (z_scores > threshold).sum()
        return 0
    
    def _handle_outliers_iqr(self, col: str, threshold: float = 1.5) -> None:
        """
        Handle outliers using IQR method (capping).
        """
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        self.df[col] = self.df[col].clip(lower, upper)
    
    def _handle_outliers_zscore(self, col: str, threshold: float = 3.0) -> None:
        """
        Handle outliers using Z-score method.
        """
        mean = self.df[col].mean()
        std = self.df[col].std()
        lower = mean - threshold * std
        upper = mean + threshold * std
        self.df[col] = self.df[col].clip(lower, upper)
    
    def _handle_outliers_percentile(self, col: str, lower_pct: float = 0.01, upper_pct: float = 0.99) -> None:
        """
        Handle outliers using percentile capping.
        """
        lower = self.df[col].quantile(lower_pct)
        upper = self.df[col].quantile(upper_pct)
        self.df[col] = self.df[col].clip(lower, upper)
    
    def remove_duplicates(
        self, 
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> 'DataCleaner':
        """
        Remove duplicate records.
        
        Args:
            subset: Columns to consider for duplicate detection
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            Self for method chaining
        """
        logger.info("Removing duplicate records...")
        
        initial_count = len(self.df)
        
        if subset is None:
            subset = ['customer_id']
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        removed = initial_count - len(self.df)
        self.cleaning_report['duplicates_removed'] = removed
        
        logger.info(f"Removed {removed} duplicate records")
        return self
    
    def fix_data_types(self) -> 'DataCleaner':
        """
        Ensure correct data types for all columns.
        
        Returns:
            Self for method chaining
        """
        logger.info("Fixing data types...")
        
        # Define expected types
        numeric_cols = [
            'age', 'employment_length', 'income', 'credit_score',
            'num_credit_lines', 'total_credit_limit', 'credit_balance',
            'credit_utilization', 'monthly_debt', 'dti_ratio',
            'delinquencies_2yr', 'bankruptcies', 'loan_amount',
            'loan_term', 'interest_rate', 'monthly_payment',
            'loan_to_income', 'default'
        ]
        
        categorical_cols = [
            'home_ownership', 'marital_status', 'education', 'loan_purpose'
        ]
        
        date_cols = ['application_date']
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert categorical columns
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # Convert date columns
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        logger.info("Data types fixed")
        return self
    
    def validate_ranges(self) -> 'DataCleaner':
        """
        Validate and correct value ranges for known columns.
        
        Returns:
            Self for method chaining
        """
        logger.info("Validating value ranges...")
        
        # Define valid ranges
        range_constraints = {
            'age': (18, 100),
            'employment_length': (0, 50),
            'credit_score': (300, 850),
            'credit_utilization': (0, 100),
            'dti_ratio': (0, 100),
            'interest_rate': (0, 50),
            'loan_to_income': (0, 200),
            'delinquencies_2yr': (0, 20),
            'bankruptcies': (0, 5),
            'default': (0, 1)
        }
        
        for col, (min_val, max_val) in range_constraints.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(min_val, max_val)
        
        logger.info("Value ranges validated")
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            Cleaned DataFrame
        """
        self.cleaning_report['final_rows'] = len(self.df)
        self.cleaning_report['final_cols'] = len(self.df.columns)
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        """
        Get the cleaning report.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.cleaning_report
    
    def print_report(self) -> None:
        """
        Print a formatted cleaning report.
        """
        report = self.cleaning_report
        print("\n" + "="*60)
        print("DATA CLEANING REPORT")
        print("="*60)
        print(f"Initial rows: {report['initial_rows']:,}")
        print(f"Initial columns: {report['initial_cols']}")
        print(f"\nMissing Values:")
        print(f"  Before: {report['missing_handled'].get('before', 'N/A')}")
        print(f"  After: {report['missing_handled'].get('after', 'N/A')}")
        print(f"  Strategy: {report['missing_handled'].get('strategy', 'N/A')}")
        print(f"\nDuplicates removed: {report['duplicates_removed']}")
        print(f"\nOutliers handled in: {len(report['outliers_handled'])} columns")
        print(f"\nFinal rows: {report['final_rows']:,}")
        print(f"Final columns: {report['final_cols']}")
        print("="*60 + "\n")


def clean_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(df)
    
    cleaned_df = (
        cleaner
        .fix_data_types()
        .remove_duplicates()
        .handle_missing_values(strategy='smart')
        .handle_outliers(method='iqr', threshold=1.5)
        .validate_ranges()
        .get_cleaned_data()
    )
    
    cleaner.print_report()
    return cleaned_df


def main():
    """
    Main function to run data cleaning.
    """
    print("="*60)
    print("CREDIT RISK PREDICTION - DATA CLEANING")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load raw data
    raw_data_path = os.path.join(RAW_DATA_DIR, 'credit_data.csv')
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data not found at {raw_data_path}")
        logger.info("Please run data_ingestion.py first")
        return None
    
    df = load_data(raw_data_path)
    print_dataframe_info(df, "Raw Data")
    
    # Run cleaning pipeline
    cleaned_df = clean_data_pipeline(df)
    print_dataframe_info(cleaned_df, "Cleaned Data")
    
    # Save cleaned data
    output_path = os.path.join(PROCESSED_DATA_DIR, 'credit_data_cleaned.csv')
    save_data(cleaned_df, output_path)
    
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Total records: {len(cleaned_df):,}")
    
    return cleaned_df


if __name__ == "__main__":
    main()
