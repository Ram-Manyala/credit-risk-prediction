"""
Data Ingestion Module for Credit Risk Prediction

This module handles:
- Generating synthetic credit risk data for demonstration
- Loading data from various sources (CSV, Database, API)
- Initial data validation and quality checks
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple

from .utils import (
    create_directories, save_data, RAW_DATA_DIR, 
    print_dataframe_info, logger
)

# Set random seed for reproducibility
np.random.seed(42)


class CreditDataGenerator:
    """
    Generates synthetic credit risk data for model development.
    
    The generated data mimics real-world credit data patterns including
    correlations between features and realistic default rates.
    """
    
    def __init__(self, n_samples: int = 10000):
        """
        Initialize the data generator.
        
        Args:
            n_samples: Number of samples to generate
        """
        self.n_samples = n_samples
        self.data = None
        
    def generate_customer_demographics(self) -> pd.DataFrame:
        """
        Generate customer demographic features.
        
        Returns:
            DataFrame with demographic features
        """
        n = self.n_samples
        
        # Age distribution (18-75, skewed towards 30-50)
        age = np.clip(
            np.random.normal(40, 12, n),
            18, 75
        ).astype(int)
        
        # Employment length (correlated with age)
        max_emp_length = np.maximum(0, age - 18)
        employment_length = np.array([
            np.random.randint(0, max(1, mel)) 
            for mel in max_emp_length
        ])
        
        # Home ownership (categorical)
        home_ownership_probs = [0.35, 0.40, 0.20, 0.05]  # MORTGAGE, RENT, OWN, OTHER
        home_ownership = np.random.choice(
            ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
            n,
            p=home_ownership_probs
        )
        
        # Marital status
        marital_status = np.random.choice(
            ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED'],
            n,
            p=[0.45, 0.35, 0.15, 0.05]
        )
        
        # Education level
        education = np.random.choice(
            ['HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'PHD', 'OTHER'],
            n,
            p=[0.30, 0.40, 0.20, 0.05, 0.05]
        )
        
        return pd.DataFrame({
            'age': age,
            'employment_length': employment_length,
            'home_ownership': home_ownership,
            'marital_status': marital_status,
            'education': education
        })
    
    def generate_financial_features(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Generate financial features correlated with demographics.
        
        Args:
            demographics: DataFrame with demographic features
            
        Returns:
            DataFrame with financial features
        """
        n = self.n_samples
        age = demographics['age'].values
        emp_length = demographics['employment_length'].values
        education = demographics['education'].values
        
        # Base income (correlated with age, employment, education)
        education_multiplier = np.where(
            education == 'PHD', 1.5,
            np.where(education == 'MASTERS', 1.3,
            np.where(education == 'BACHELORS', 1.1, 1.0))
        )
        
        base_income = 30000 + (age * 500) + (emp_length * 2000)
        income = (base_income * education_multiplier * 
                  np.random.uniform(0.7, 1.3, n)).astype(int)
        income = np.clip(income, 15000, 500000)
        
        # Credit score (300-850, correlated with income and employment)
        credit_score_base = 500 + (income / 2000) + (emp_length * 5)
        credit_score = np.clip(
            credit_score_base + np.random.normal(0, 50, n),
            300, 850
        ).astype(int)
        
        # Number of credit lines
        num_credit_lines = np.clip(
            np.random.poisson(3, n) + (credit_score - 600) // 50,
            0, 20
        ).astype(int)
        
        # Total credit limit
        total_credit_limit = (
            (credit_score * 50) + 
            (income * 0.3) + 
            np.random.normal(0, 5000, n)
        ).astype(int)
        total_credit_limit = np.clip(total_credit_limit, 1000, 200000)
        
        # Current credit balance (0% - 100% of limit)
        utilization_tendency = np.random.beta(2, 5, n)  # Skewed towards lower utilization
        credit_balance = (total_credit_limit * utilization_tendency).astype(int)
        
        # Calculate credit utilization
        credit_utilization = np.round(credit_balance / total_credit_limit * 100, 2)
        
        # Monthly debt payments
        monthly_debt = (credit_balance * 0.03 + 
                       np.random.uniform(100, 500, n)).astype(int)
        
        # DTI ratio
        monthly_income = income / 12
        dti_ratio = np.round(monthly_debt / monthly_income * 100, 2)
        dti_ratio = np.clip(dti_ratio, 0, 100)
        
        # Delinquencies in last 2 years (inversely correlated with credit score)
        delinquency_prob = np.clip((800 - credit_score) / 1000, 0.01, 0.5)
        delinquencies_2yr = np.array([
            np.random.binomial(5, p) for p in delinquency_prob
        ])
        
        # Bankruptcies
        bankruptcy_prob = np.where(credit_score < 550, 0.1, 0.02)
        bankruptcies = np.random.binomial(1, bankruptcy_prob)
        
        return pd.DataFrame({
            'income': income,
            'credit_score': credit_score,
            'num_credit_lines': num_credit_lines,
            'total_credit_limit': total_credit_limit,
            'credit_balance': credit_balance,
            'credit_utilization': credit_utilization,
            'monthly_debt': monthly_debt,
            'dti_ratio': dti_ratio,
            'delinquencies_2yr': delinquencies_2yr,
            'bankruptcies': bankruptcies
        })
    
    def generate_loan_features(self, financial: pd.DataFrame) -> pd.DataFrame:
        """
        Generate loan-specific features.
        
        Args:
            financial: DataFrame with financial features
            
        Returns:
            DataFrame with loan features
        """
        n = self.n_samples
        income = financial['income'].values
        credit_score = financial['credit_score'].values
        
        # Loan purpose
        loan_purpose = np.random.choice(
            ['DEBT_CONSOLIDATION', 'CREDIT_CARD', 'HOME_IMPROVEMENT', 
             'MAJOR_PURCHASE', 'MEDICAL', 'VACATION', 'BUSINESS', 'OTHER'],
            n,
            p=[0.35, 0.20, 0.15, 0.10, 0.07, 0.05, 0.05, 0.03]
        )
        
        # Loan amount (based on income and credit score)
        max_loan = income * 0.5 + (credit_score - 600) * 100
        loan_amount = np.clip(
            np.random.uniform(1000, max_loan, n),
            1000, 100000
        ).astype(int)
        
        # Loan term (36 or 60 months typically)
        loan_term = np.random.choice([36, 60], n, p=[0.6, 0.4])
        
        # Interest rate (based on credit score)
        base_rate = 25 - (credit_score - 300) / 50
        interest_rate = np.clip(
            base_rate + np.random.normal(0, 2, n),
            5, 30
        )
        interest_rate = np.round(interest_rate, 2)
        
        # Monthly payment
        monthly_rate = interest_rate / 100 / 12
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** loan_term
        ) / ((1 + monthly_rate) ** loan_term - 1)
        monthly_payment = np.round(monthly_payment, 2)
        
        # Loan to income ratio
        loan_to_income = np.round(loan_amount / income * 100, 2)
        
        return pd.DataFrame({
            'loan_purpose': loan_purpose,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'interest_rate': interest_rate,
            'monthly_payment': monthly_payment,
            'loan_to_income': loan_to_income
        })
    
    def generate_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate target variable (default) based on risk factors.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Array of binary default indicators
        """
        # Calculate default probability based on risk factors
        risk_score = (
            -0.005 * df['credit_score'] +
            0.02 * df['dti_ratio'] +
            0.03 * df['credit_utilization'] +
            0.1 * df['delinquencies_2yr'] +
            0.5 * df['bankruptcies'] +
            0.01 * df['interest_rate'] +
            0.001 * df['loan_to_income'] +
            -0.01 * df['employment_length']
        )
        
        # Convert to probability using logistic function
        default_prob = 1 / (1 + np.exp(-risk_score))
        
        # Adjust to achieve ~15% default rate
        default_prob = np.clip(default_prob * 0.8, 0.02, 0.95)
        
        # Generate binary outcome
        default = np.random.binomial(1, default_prob)
        
        return default
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete credit risk dataset.
        
        Returns:
            Complete DataFrame with all features and target
        """
        logger.info(f"Generating synthetic dataset with {self.n_samples:,} samples...")
        
        # Generate customer IDs
        customer_ids = [f'CUST_{i:06d}' for i in range(1, self.n_samples + 1)]
        
        # Generate application dates
        start_date = datetime(2022, 1, 1)
        date_range = 730  # 2 years of data
        application_dates = [
            start_date + timedelta(days=np.random.randint(0, date_range))
            for _ in range(self.n_samples)
        ]
        
        # Generate all features
        demographics = self.generate_customer_demographics()
        financial = self.generate_financial_features(demographics)
        loan = self.generate_loan_features(financial)
        
        # Combine all features
        self.data = pd.concat([demographics, financial, loan], axis=1)
        self.data.insert(0, 'customer_id', customer_ids)
        self.data.insert(1, 'application_date', application_dates)
        
        # Generate target
        self.data['default'] = self.generate_target(self.data)
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"Default rate: {self.data['default'].mean()*100:.2f}%")
        
        return self.data


class DataIngestion:
    """
    Handles data ingestion from various sources.
    """
    
    def __init__(self):
        create_directories()
        
    def ingest_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Ingest data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with the ingested data
        """
        logger.info(f"Ingesting data from {filepath}")
        df = pd.read_csv(filepath)
        self._validate_data(df)
        return df
    
    def ingest_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate and ingest synthetic credit data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        generator = CreditDataGenerator(n_samples)
        df = generator.generate_dataset()
        
        # Save to raw data directory
        output_path = os.path.join(RAW_DATA_DIR, 'credit_data.csv')
        save_data(df, output_path)
        logger.info(f"Synthetic data saved to {output_path}")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Perform basic data validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        logger.info("Performing data validation...")
        
        # Check for empty dataframe
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for required columns
        required_columns = ['customer_id', 'default']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing recommended columns: {missing_cols}")
        
        # Check for duplicate IDs
        if 'customer_id' in df.columns:
            duplicates = df['customer_id'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate customer IDs")
        
        # Check target variable
        if 'default' in df.columns:
            unique_values = df['default'].unique()
            if not set(unique_values).issubset({0, 1}):
                logger.warning(f"Target variable has unexpected values: {unique_values}")
        
        logger.info("Data validation completed")
        return True


def main():
    """
    Main function to run data ingestion.
    """
    print("="*60)
    print("CREDIT RISK PREDICTION - DATA INGESTION")
    print("="*60)
    
    # Initialize ingestion
    ingestion = DataIngestion()
    
    # Generate synthetic data
    df = ingestion.ingest_synthetic_data(n_samples=10000)
    
    # Print summary
    print_dataframe_info(df, "Credit Risk Dataset")
    
    print("\n" + "="*60)
    print("DATA INGESTION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nData saved to: {os.path.join(RAW_DATA_DIR, 'credit_data.csv')}")
    print(f"Total records: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Default rate: {df['default'].mean()*100:.2f}%")
    
    return df


if __name__ == "__main__":
    main()
