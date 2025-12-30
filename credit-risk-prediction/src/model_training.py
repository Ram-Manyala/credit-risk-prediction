"""
Model Training Module for Credit Risk Prediction

This module handles:
- Data preparation for modeling
- Training multiple ML models
- Hyperparameter tuning
- Model selection and persistence
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

from .utils import (
    load_data, save_model, FEATURES_DIR, MODELS_DIR,
    print_dataframe_info, logger, create_directories,
    get_feature_importance_df
)


class CreditRiskModelTrainer:
    """
    Comprehensive model training pipeline for credit risk prediction.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'default',
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self) -> 'CreditRiskModelTrainer':
        """
        Prepare data for modeling.
        
        Returns:
            Self for method chaining
        """
        logger.info("Preparing data for modeling...")
        
        # Exclude non-feature columns
        exclude_cols = ['customer_id', 'application_date', self.target_col]
        
        # Get feature columns (only numeric)
        feature_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols 
            and self.df[col].dtype in ['int64', 'float64']
        ]
        
        self.feature_names = feature_cols
        self.X = self.df[feature_cols].values
        self.y = self.df[self.target_col].values
        
        # Handle any remaining NaN values
        self.X = np.nan_to_num(self.X, nan=0)
        
        logger.info(f"Features shape: {self.X.shape}")
        logger.info(f"Target distribution: {np.bincount(self.y.astype(int))}")
        
        return self
    
    def split_data(self) -> 'CreditRiskModelTrainer':
        """
        Split data into training and test sets.
        
        Returns:
            Self for method chaining
        """
        logger.info("Splitting data into train and test sets...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        logger.info(f"Training default rate: {self.y_train.mean()*100:.2f}%")
        logger.info(f"Test default rate: {self.y_test.mean()*100:.2f}%")
        
        return self
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model names and instances
        """
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=sum(self.y_train == 0) / sum(self.y_train == 1)
            )
        
        return models
    
    def train_models(self, use_cv: bool = True, cv_folds: int = 5) -> 'CreditRiskModelTrainer':
        """
        Train all models.
        
        Args:
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Self for method chaining
        """
        logger.info("Training models...")
        
        models = self.get_models()
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, zero_division=0),
                    'f1': f1_score(self.y_test, y_pred, zero_division=0),
                    'auc_roc': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                # Cross-validation score
                if use_cv:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                        random_state=self.random_state)
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train,
                        cv=cv, scoring='roc_auc'
                    )
                    metrics['cv_auc_mean'] = cv_scores.mean()
                    metrics['cv_auc_std'] = cv_scores.std()
                
                # Calculate Gini coefficient
                metrics['gini'] = 2 * metrics['auc_roc'] - 1
                
                self.models[name] = model
                self.results[name] = metrics
                
                logger.info(f"{name} - AUC: {metrics['auc_roc']:.4f}, "
                           f"F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        return self
    
    def tune_best_model(
        self,
        model_name: Optional[str] = None,
        param_grid: Optional[Dict] = None
    ) -> 'CreditRiskModelTrainer':
        """
        Tune hyperparameters for the best model.
        
        Args:
            model_name: Name of model to tune (None = auto-select best)
            param_grid: Custom parameter grid
            
        Returns:
            Self for method chaining
        """
        logger.info("Tuning best model hyperparameters...")
        
        # Auto-select best model if not specified
        if model_name is None:
            model_name = max(self.results, key=lambda x: self.results[x]['auc_roc'])
        
        logger.info(f"Tuning {model_name}...")
        
        # Default parameter grids
        default_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 150],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.05, 0.1],
                'min_samples_split': [5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 150],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        }
        
        if param_grid is None:
            param_grid = default_grids.get(model_name, {})
        
        if not param_grid:
            logger.info(f"No parameter grid for {model_name}, skipping tuning")
            return self
        
        # Get base model
        base_model = self.get_models()[model_name]
        
        # Grid search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        tuned_model = grid_search.best_estimator_
        
        # Evaluate tuned model
        y_pred = tuned_model.predict(self.X_test)
        y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1]
        
        tuned_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'best_params': grid_search.best_params_
        }
        tuned_metrics['gini'] = 2 * tuned_metrics['auc_roc'] - 1
        
        # Update models and results
        tuned_name = f"{model_name} (Tuned)"
        self.models[tuned_name] = tuned_model
        self.results[tuned_name] = tuned_metrics
        
        logger.info(f"Tuned {model_name} - AUC: {tuned_metrics['auc_roc']:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return self
    
    def select_best_model(self) -> Tuple[str, Any]:
        """
        Select the best performing model.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        logger.info("Selecting best model...")
        
        self.best_model_name = max(
            self.results, 
            key=lambda x: self.results[x]['auc_roc']
        )
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Best AUC-ROC: {self.results[self.best_model_name]['auc_roc']:.4f}")
        
        return self.best_model_name, self.best_model
    
    def save_best_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the best model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            self.select_best_model()
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'best_model.pkl')
        
        # Save model along with feature names and metadata
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'metrics': self.results[self.best_model_name]
        }
        
        save_model(model_package, filepath)
        logger.info(f"Best model saved to {filepath}")
        
        return filepath
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            self.select_best_model()
        
        model = self.best_model
        
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            if 'classifier' in model.named_steps:
                model = model.named_steps['classifier']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = get_feature_importance_df(
                model, self.feature_names
            )
            return importance_df
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
            importance_df['importance_pct'] = (
                importance_df['importance'] / importance_df['importance'].sum() * 100
            )
            return importance_df
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
    
    def print_results(self) -> None:
        """
        Print formatted model comparison results.
        """
        print("\n" + "="*80)
        print("MODEL TRAINING RESULTS")
        print("="*80)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\nModel Comparison:")
        print("-"*80)
        
        display_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'gini']
        available_cols = [col for col in display_cols if col in results_df.columns]
        
        print(results_df[available_cols].to_string())
        
        print("\n" + "-"*80)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best AUC-ROC: {self.results[self.best_model_name]['auc_roc']:.4f}")
        print(f"Best Gini: {self.results[self.best_model_name]['gini']:.4f}")
        print("="*80 + "\n")


def train_models_pipeline(df: pd.DataFrame) -> Tuple[Any, Dict]:
    """
    Run the complete model training pipeline.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Tuple of (best_model, results_dict)
    """
    trainer = CreditRiskModelTrainer(df)
    
    # Run training pipeline
    trainer.prepare_data()
    trainer.split_data()
    trainer.train_models(use_cv=True)
    trainer.tune_best_model()
    trainer.select_best_model()
    
    # Print results
    trainer.print_results()
    
    # Get feature importance
    importance_df = trainer.get_feature_importance()
    if not importance_df.empty:
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Save best model
    trainer.save_best_model()
    
    return trainer.best_model, trainer.results


def main():
    """
    Main function to run model training.
    """
    print("="*60)
    print("CREDIT RISK PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load feature data
    features_path = os.path.join(FEATURES_DIR, 'credit_features.csv')
    
    if not os.path.exists(features_path):
        logger.error(f"Features not found at {features_path}")
        logger.info("Please run feature_engineering.py first")
        return None
    
    df = load_data(features_path)
    print_dataframe_info(df, "Features Data")
    
    # Run training pipeline
    best_model, results = train_models_pipeline(df)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nBest model saved to: {os.path.join(MODELS_DIR, 'best_model.pkl')}")
    
    return best_model, results


if __name__ == "__main__":
    main()
