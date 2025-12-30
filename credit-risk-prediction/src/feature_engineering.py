"""
Feature Engineering Module for Credit Risk Prediction

This module handles:
- Creating new features from existing data
- Feature transformations
- Encoding categorical variables
- Feature scaling and normalization
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

from .utils import (
    load_data, save_data, PROCESSED_DATA_DIR, FEATURES_DIR,
    print_dataframe_info, logger, create_directories
)


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for credit risk prediction.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer.
        
        Args:
            df: Cleaned DataFrame for feature engineering
        """
        self.df = df.copy()
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        self.feature_report = {
            'original_features': list(df.columns),
            'new_features': [],
            'dropped_features': [],
            'transformations': []
        }
    
    def create_risk_indicators(self) -> 'FeatureEngineer':
        """
        Create risk indicator features.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating risk indicator features...")
        
        # High DTI indicator
        self.df['high_dti'] = (self.df['dti_ratio'] > 40).astype(int)
        
        # High credit utilization indicator
        self.df['high_utilization'] = (self.df['credit_utilization'] > 70).astype(int)
        
        # Low credit score indicator
        self.df['low_credit_score'] = (self.df['credit_score'] < 650).astype(int)
        
        # Has delinquencies indicator
        self.df['has_delinquencies'] = (self.df['delinquencies_2yr'] > 0).astype(int)
        
        # Has bankruptcies indicator  
        self.df['has_bankruptcies'] = (self.df['bankruptcies'] > 0).astype(int)
        
        # New borrower (short employment)
        self.df['new_borrower'] = (self.df['employment_length'] < 2).astype(int)
        
        # High interest rate indicator
        self.df['high_interest'] = (self.df['interest_rate'] > 15).astype(int)
        
        # Large loan indicator
        self.df['large_loan'] = (self.df['loan_amount'] > 30000).astype(int)
        
        new_features = [
            'high_dti', 'high_utilization', 'low_credit_score',
            'has_delinquencies', 'has_bankruptcies', 'new_borrower',
            'high_interest', 'large_loan'
        ]
        self.feature_report['new_features'].extend(new_features)
        
        logger.info(f"Created {len(new_features)} risk indicator features")
        return self
    
    def create_financial_ratios(self) -> 'FeatureEngineer':
        """
        Create financial ratio features.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating financial ratio features...")
        
        # Payment to income ratio
        self.df['payment_to_income'] = (
            self.df['monthly_payment'] / (self.df['income'] / 12) * 100
        )
        self.df['payment_to_income'] = self.df['payment_to_income'].clip(0, 100)
        
        # Available credit ratio
        self.df['available_credit_ratio'] = (
            (self.df['total_credit_limit'] - self.df['credit_balance']) / 
            self.df['total_credit_limit'] * 100
        )
        self.df['available_credit_ratio'] = self.df['available_credit_ratio'].clip(0, 100)
        
        # Credit per credit line
        self.df['credit_per_line'] = np.where(
            self.df['num_credit_lines'] > 0,
            self.df['total_credit_limit'] / self.df['num_credit_lines'],
            0
        )
        
        # Balance per credit line
        self.df['balance_per_line'] = np.where(
            self.df['num_credit_lines'] > 0,
            self.df['credit_balance'] / self.df['num_credit_lines'],
            0
        )
        
        # Total debt to credit ratio
        self.df['debt_to_credit'] = np.where(
            self.df['total_credit_limit'] > 0,
            (self.df['credit_balance'] + self.df['loan_amount']) / 
            self.df['total_credit_limit'] * 100,
            0
        )
        self.df['debt_to_credit'] = self.df['debt_to_credit'].clip(0, 500)
        
        new_features = [
            'payment_to_income', 'available_credit_ratio',
            'credit_per_line', 'balance_per_line', 'debt_to_credit'
        ]
        self.feature_report['new_features'].extend(new_features)
        
        logger.info(f"Created {len(new_features)} financial ratio features")
        return self
    
    def create_interaction_features(self) -> 'FeatureEngineer':
        """
        Create interaction features between existing variables.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating interaction features...")
        
        # Credit score x DTI interaction
        self.df['score_dti_interaction'] = (
            (850 - self.df['credit_score']) * self.df['dti_ratio'] / 100
        )
        
        # Age x Employment stability
        self.df['age_employment_ratio'] = np.where(
            self.df['age'] > 18,
            self.df['employment_length'] / (self.df['age'] - 18) * 100,
            0
        )
        self.df['age_employment_ratio'] = self.df['age_employment_ratio'].clip(0, 100)
        
        # Credit score x Utilization interaction
        self.df['score_util_interaction'] = (
            (850 - self.df['credit_score']) * self.df['credit_utilization'] / 100
        )
        
        # Income x Loan amount interaction
        self.df['income_loan_ratio'] = self.df['income'] / (self.df['loan_amount'] + 1)
        
        # Delinquency severity score
        self.df['delinquency_severity'] = (
            self.df['delinquencies_2yr'] * 2 + self.df['bankruptcies'] * 5
        )
        
        new_features = [
            'score_dti_interaction', 'age_employment_ratio',
            'score_util_interaction', 'income_loan_ratio',
            'delinquency_severity'
        ]
        self.feature_report['new_features'].extend(new_features)
        
        logger.info(f"Created {len(new_features)} interaction features")
        return self
    
    def create_risk_score(self) -> 'FeatureEngineer':
        """
        Create a composite risk score feature.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating composite risk score...")
        
        # Normalize components to 0-1 scale
        credit_score_norm = (850 - self.df['credit_score']) / 550  # Inverted
        dti_norm = self.df['dti_ratio'] / 100
        util_norm = self.df['credit_utilization'] / 100
        delinq_norm = self.df['delinquencies_2yr'] / 5
        interest_norm = self.df['interest_rate'] / 30
        
        # Weighted risk score
        self.df['composite_risk_score'] = (
            0.30 * credit_score_norm +
            0.20 * dti_norm +
            0.20 * util_norm +
            0.20 * delinq_norm.clip(0, 1) +
            0.10 * interest_norm
        ) * 100
        
        self.df['composite_risk_score'] = self.df['composite_risk_score'].clip(0, 100)
        
        # Risk category
        self.df['risk_category'] = pd.cut(
            self.df['composite_risk_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        new_features = ['composite_risk_score', 'risk_category']
        self.feature_report['new_features'].extend(new_features)
        
        logger.info("Created composite risk score feature")
        return self
    
    def create_binned_features(self) -> 'FeatureEngineer':
        """
        Create binned versions of continuous features.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating binned features...")
        
        # Age bins
        self.df['age_group'] = pd.cut(
            self.df['age'],
            bins=[18, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Income bins
        self.df['income_bracket'] = pd.cut(
            self.df['income'],
            bins=[0, 30000, 50000, 75000, 100000, 150000, float('inf')],
            labels=['<30K', '30-50K', '50-75K', '75-100K', '100-150K', '150K+']
        )
        
        # Credit score bins
        self.df['credit_tier'] = pd.cut(
            self.df['credit_score'],
            bins=[300, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Loan amount bins
        self.df['loan_size'] = pd.cut(
            self.df['loan_amount'],
            bins=[0, 5000, 15000, 30000, 50000, float('inf')],
            labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        )
        
        new_features = ['age_group', 'income_bracket', 'credit_tier', 'loan_size']
        self.feature_report['new_features'].extend(new_features)
        
        logger.info(f"Created {len(new_features)} binned features")
        return self
    
    def encode_categorical(self, method: str = 'label') -> 'FeatureEngineer':
        """
        Encode categorical variables.
        
        Args:
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Encoding categorical variables with method: {method}")
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols 
                           if col not in ['customer_id', 'application_date']]
        
        if method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(
                    self.df[col].astype(str)
                )
                self.encoders[col] = le
                self.feature_report['transformations'].append(
                    f"{col} -> {col}_encoded (label encoding)"
                )
        elif method == 'onehot':
            for col in categorical_cols:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.feature_report['transformations'].append(
                    f"{col} -> one-hot encoded ({len(dummies.columns)} columns)"
                )
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return self
    
    def scale_features(
        self, 
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> 'FeatureEngineer':
        """
        Scale numeric features.
        
        Args:
            columns: Specific columns to scale (None = all numeric)
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Scaling features with method: {method}")
        
        if columns is None:
            exclude_cols = [
                'customer_id', 'application_date', 'default',
                'high_dti', 'high_utilization', 'low_credit_score',
                'has_delinquencies', 'has_bankruptcies', 'new_borrower',
                'high_interest', 'large_loan'
            ]
            columns = [
                col for col in self.df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols and '_encoded' not in col
            ]
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[columns])
        
        for i, col in enumerate(columns):
            self.df[f'{col}_scaled'] = scaled_data[:, i]
        
        self.scalers['standard'] = scaler
        self.feature_report['transformations'].append(
            f"Scaled {len(columns)} features using {method} scaling"
        )
        
        logger.info(f"Scaled {len(columns)} features")
        return self
    
    def select_features_for_modeling(self) -> pd.DataFrame:
        """
        Select final features for modeling.
        
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features for modeling...")
        
        # Features to keep for modeling
        modeling_features = [
            # Original numeric features
            'age', 'employment_length', 'income', 'credit_score',
            'num_credit_lines', 'total_credit_limit', 'credit_balance',
            'credit_utilization', 'monthly_debt', 'dti_ratio',
            'delinquencies_2yr', 'bankruptcies', 'loan_amount',
            'loan_term', 'interest_rate', 'monthly_payment', 'loan_to_income',
            
            # Risk indicators
            'high_dti', 'high_utilization', 'low_credit_score',
            'has_delinquencies', 'has_bankruptcies', 'new_borrower',
            'high_interest', 'large_loan',
            
            # Financial ratios
            'payment_to_income', 'available_credit_ratio',
            'credit_per_line', 'balance_per_line', 'debt_to_credit',
            
            # Interaction features
            'score_dti_interaction', 'age_employment_ratio',
            'score_util_interaction', 'income_loan_ratio',
            'delinquency_severity',
            
            # Risk score
            'composite_risk_score',
            
            # Encoded categoricals
            'home_ownership_encoded', 'marital_status_encoded',
            'education_encoded', 'loan_purpose_encoded'
        ]
        
        # Add target
        modeling_features.append('default')
        
        # Keep customer_id for reference
        id_cols = ['customer_id']
        
        # Filter to available columns
        available_features = [
            col for col in modeling_features 
            if col in self.df.columns
        ]
        
        final_cols = id_cols + available_features
        
        logger.info(f"Selected {len(available_features)} features for modeling")
        return self.df[final_cols]
    
    def get_feature_report(self) -> Dict:
        """
        Get the feature engineering report.
        
        Returns:
            Dictionary with feature engineering details
        """
        return self.feature_report
    
    def print_report(self) -> None:
        """
        Print a formatted feature engineering report.
        """
        report = self.feature_report
        print("\n" + "="*60)
        print("FEATURE ENGINEERING REPORT")
        print("="*60)
        print(f"\nOriginal features: {len(report['original_features'])}")
        print(f"New features created: {len(report['new_features'])}")
        print(f"\nNew Features:")
        for feat in report['new_features']:
            print(f"  - {feat}")
        print(f"\nTransformations:")
        for trans in report['transformations']:
            print(f"  - {trans}")
        print(f"\nFinal feature count: {len(self.df.columns)}")
        print("="*60 + "\n")


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(df)
    
    # Run feature engineering steps
    engineer.create_risk_indicators()
    engineer.create_financial_ratios()
    engineer.create_interaction_features()
    engineer.create_risk_score()
    engineer.create_binned_features()
    engineer.encode_categorical(method='label')
    
    # Get final features for modeling
    features_df = engineer.select_features_for_modeling()
    
    engineer.print_report()
    return features_df


def main():
    """
    Main function to run feature engineering.
    """
    print("="*60)
    print("CREDIT RISK PREDICTION - FEATURE ENGINEERING")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load cleaned data
    cleaned_data_path = os.path.join(PROCESSED_DATA_DIR, 'credit_data_cleaned.csv')
    
    if not os.path.exists(cleaned_data_path):
        logger.error(f"Cleaned data not found at {cleaned_data_path}")
        logger.info("Please run data_cleaning.py first")
        return None
    
    df = load_data(cleaned_data_path)
    print_dataframe_info(df, "Cleaned Data")
    
    # Run feature engineering pipeline
    features_df = feature_engineering_pipeline(df)
    print_dataframe_info(features_df, "Engineered Features")
    
    # Save engineered features
    output_path = os.path.join(FEATURES_DIR, 'credit_features.csv')
    save_data(features_df, output_path)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nFeatures saved to: {output_path}")
    print(f"Total records: {len(features_df):,}")
    print(f"Total features: {len(features_df.columns) - 2}")  # Exclude ID and target
    
    return features_df


if __name__ == "__main__":
    main()
