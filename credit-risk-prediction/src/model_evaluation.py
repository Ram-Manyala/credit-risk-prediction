"""
Model Evaluation Module for Credit Risk Prediction

This module handles:
- Comprehensive model evaluation
- Performance visualization
- Generating evaluation reports
- Model comparison analysis
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

from .utils import (
    load_data, load_model, FEATURES_DIR, MODELS_DIR,
    FIGURES_DIR, REPORTS_DIR, logger, create_directories,
    generate_report
)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class ModelEvaluator:
    """
    Comprehensive model evaluation for credit risk models.
    """
    
    def __init__(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        model_name: str = 'Credit Risk Model'
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            model_name: Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model_name = model_name
        
        # Generate predictions
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        self.metrics = {}
        self.figures = {}
        
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        logger.info("Calculating evaluation metrics...")
        
        self.metrics = {
            # Classification metrics
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, self.y_pred, zero_division=0),
            
            # Probability metrics
            'auc_roc': roc_auc_score(self.y_test, self.y_pred_proba),
            'average_precision': average_precision_score(self.y_test, self.y_pred_proba),
            'brier_score': brier_score_loss(self.y_test, self.y_pred_proba),
            
            # Derived metrics
            'gini_coefficient': 2 * roc_auc_score(self.y_test, self.y_pred_proba) - 1,
        }
        
        # Calculate KS statistic
        self.metrics['ks_statistic'] = self._calculate_ks_statistic()
        
        # Calculate lift metrics
        lift_metrics = self._calculate_lift_metrics()
        self.metrics.update(lift_metrics)
        
        logger.info(f"AUC-ROC: {self.metrics['auc_roc']:.4f}")
        logger.info(f"Gini: {self.metrics['gini_coefficient']:.4f}")
        
        return self.metrics
    
    def _calculate_ks_statistic(self) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Returns:
            KS statistic value
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        ks_stat = max(tpr - fpr)
        return ks_stat
    
    def _calculate_lift_metrics(
        self, 
        top_percentile: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate lift metrics.
        
        Args:
            top_percentile: Top percentile for lift calculation
            
        Returns:
            Dictionary of lift metrics
        """
        # Sort by predicted probability
        sorted_idx = np.argsort(self.y_pred_proba)[::-1]
        n_top = int(len(self.y_test) * top_percentile)
        
        # Calculate lift
        baseline_rate = self.y_test.mean()
        top_rate = self.y_test[sorted_idx[:n_top]].mean()
        
        lift = top_rate / baseline_rate if baseline_rate > 0 else 0
        
        return {
            'lift_at_10': lift,
            'capture_rate_at_10': self.y_test[sorted_idx[:n_top]].sum() / self.y_test.sum()
        }
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.model_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add Gini annotation
        gini = 2 * roc_auc - 1
        ax.annotate(f'Gini = {gini:.3f}', xy=(0.6, 0.2),
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        self.figures['roc_curve'] = fig
        return fig
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting Precision-Recall curve...")
        
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = self.y_test.mean()
        ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                  label=f'Baseline (prevalence = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.model_name}', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        self.figures['pr_curve'] = fig
        return fig
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting confusion matrix...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {self.model_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        self.figures['confusion_matrix'] = fig
        return fig
    
    def plot_probability_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot predicted probability distribution by class.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting probability distribution...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate probabilities by class
        proba_0 = self.y_pred_proba[self.y_test == 0]
        proba_1 = self.y_pred_proba[self.y_test == 1]
        
        ax.hist(proba_0, bins=50, alpha=0.6, label='No Default', color='blue')
        ax.hist(proba_1, bins=50, alpha=0.6, label='Default', color='red')
        
        ax.set_xlabel('Predicted Probability of Default', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Score Distribution - {self.model_name}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        self.figures['probability_distribution'] = fig
        return fig
    
    def plot_feature_importance(
        self, 
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting feature importance...")
        
        model = self.model
        
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            if 'classifier' in model.named_steps:
                model = model.named_steps['classifier']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't support feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values(
            'importance', ascending=True
        ).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
        ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {self.model_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        self.figures['feature_importance'] = fig
        return fig
    
    def plot_calibration_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting calibration curve...")
        
        prob_true, prob_pred = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=10
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(prob_pred, prob_true, 's-', color='darkorange', lw=2,
                label=f'{self.model_name}')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        self.figures['calibration_curve'] = fig
        return fig
    
    def generate_all_plots(self, output_dir: Optional[str] = None) -> Dict:
        """
        Generate all evaluation plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of figure names and objects
        """
        if output_dir is None:
            output_dir = FIGURES_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_roc_curve(os.path.join(output_dir, 'roc_curve.png'))
        self.plot_precision_recall_curve(os.path.join(output_dir, 'pr_curve.png'))
        self.plot_confusion_matrix(os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_probability_distribution(os.path.join(output_dir, 'score_distribution.png'))
        self.plot_feature_importance(save_path=os.path.join(output_dir, 'feature_importance.png'))
        self.plot_calibration_curve(os.path.join(output_dir, 'calibration_curve.png'))
        
        plt.close('all')
        
        return self.figures
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        report = f"""
================================================================================
                      CREDIT RISK MODEL EVALUATION REPORT
================================================================================

Model: {self.model_name}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
1. PERFORMANCE METRICS
--------------------------------------------------------------------------------

Classification Metrics:
  - Accuracy:           {self.metrics['accuracy']:.4f}
  - Precision:          {self.metrics['precision']:.4f}
  - Recall:             {self.metrics['recall']:.4f}
  - F1-Score:           {self.metrics['f1_score']:.4f}

Discriminatory Power:
  - AUC-ROC:            {self.metrics['auc_roc']:.4f}
  - Gini Coefficient:   {self.metrics['gini_coefficient']:.4f}
  - KS Statistic:       {self.metrics['ks_statistic']:.4f}

Calibration:
  - Brier Score:        {self.metrics['brier_score']:.4f}
  - Average Precision:  {self.metrics['average_precision']:.4f}

Lift Metrics:
  - Lift at 10%:        {self.metrics['lift_at_10']:.2f}x
  - Capture Rate at 10%:{self.metrics['capture_rate_at_10']*100:.1f}%

--------------------------------------------------------------------------------
2. CONFUSION MATRIX
--------------------------------------------------------------------------------

{classification_report(self.y_test, self.y_pred, target_names=['No Default', 'Default'])}

--------------------------------------------------------------------------------
3. MODEL INTERPRETATION
--------------------------------------------------------------------------------

Risk Score Interpretation:
- AUC-ROC of {self.metrics['auc_roc']:.4f} indicates {'excellent' if self.metrics['auc_roc'] > 0.8 else 'good' if self.metrics['auc_roc'] > 0.7 else 'fair'} discriminatory power
- Gini coefficient of {self.metrics['gini_coefficient']:.4f} suggests the model effectively separates defaulters from non-defaulters
- KS statistic of {self.metrics['ks_statistic']:.4f} shows {'strong' if self.metrics['ks_statistic'] > 0.4 else 'moderate'} separation between classes

Business Impact:
- At 10% cutoff, the model captures {self.metrics['capture_rate_at_10']*100:.1f}% of all defaults
- Lift of {self.metrics['lift_at_10']:.2f}x means top 10% of predictions are {self.metrics['lift_at_10']:.2f} times more likely to default than average

================================================================================
                                   END OF REPORT
================================================================================
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def save_metrics_json(self, output_path: Optional[str] = None) -> str:
        """
        Save metrics as JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path where metrics were saved
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        if output_path is None:
            output_path = os.path.join(REPORTS_DIR, 'evaluation_metrics.json')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        metrics_output = {
            'model_name': self.model_name,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'metrics': {k: float(v) for k, v in self.metrics.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_output, f, indent=4)
        
        logger.info(f"Metrics saved to {output_path}")
        return output_path


def evaluate_model_pipeline(model_path: str, data_path: str) -> Dict:
    """
    Run complete model evaluation pipeline.
    
    Args:
        model_path: Path to saved model
        data_path: Path to feature data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model package
    model_package = load_model(model_path)
    model = model_package['model']
    model_name = model_package['model_name']
    feature_names = model_package['feature_names']
    
    # Load data
    df = load_data(data_path)
    
    # Prepare test data
    exclude_cols = ['customer_id', 'application_date', 'default']
    X = df[[col for col in feature_names if col in df.columns]].values
    y = df['default'].values
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0)
    
    # Use last 20% as test set (to match training split)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=[col for col in feature_names if col in df.columns],
        model_name=model_name
    )
    
    # Run evaluation
    metrics = evaluator.calculate_all_metrics()
    evaluator.generate_all_plots()
    report = evaluator.generate_report(
        os.path.join(REPORTS_DIR, 'evaluation_report.txt')
    )
    evaluator.save_metrics_json()
    
    print(report)
    
    return metrics


def main():
    """
    Main function to run model evaluation.
    """
    print("="*60)
    print("CREDIT RISK PREDICTION - MODEL EVALUATION")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Paths
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    features_path = os.path.join(FEATURES_DIR, 'credit_features.csv')
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Please run model_training.py first")
        return None
    
    if not os.path.exists(features_path):
        logger.error(f"Features not found at {features_path}")
        logger.info("Please run feature_engineering.py first")
        return None
    
    # Run evaluation
    metrics = evaluate_model_pipeline(model_path, features_path)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")
    
    return metrics


if __name__ == "__main__":
    main()
