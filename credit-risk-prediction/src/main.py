"""
Main Pipeline Runner for Credit Risk Prediction

This script runs the complete end-to-end credit risk prediction pipeline.
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import main as run_ingestion
from src.data_cleaning import main as run_cleaning
from src.feature_engineering import main as run_feature_engineering
from src.model_training import main as run_training
from src.model_evaluation import main as run_evaluation
from src.utils import create_directories, logger


def run_complete_pipeline():
    """
    Run the complete credit risk prediction pipeline.
    """
    print("\n" + "="*80)
    print("     CREDIT RISK PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Create directories
    create_directories()
    
    steps = [
        ("Data Ingestion", run_ingestion),
        ("Data Cleaning", run_cleaning),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training", run_training),
        ("Model Evaluation", run_evaluation),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'*'*80}")
        print(f"STEP: {step_name}")
        print(f"{'*'*80}\n")
        
        step_start = time.time()
        
        try:
            result = step_func()
            results[step_name] = {
                'status': 'SUCCESS',
                'result': result,
                'duration': time.time() - step_start
            }
            logger.info(f"{step_name} completed in {results[step_name]['duration']:.2f}s")
        except Exception as e:
            results[step_name] = {
                'status': 'FAILED',
                'error': str(e),
                'duration': time.time() - step_start
            }
            logger.error(f"{step_name} failed: {str(e)}")
            print(f"\nERROR in {step_name}: {str(e)}")
            # Continue to next step or break based on criticality
            if step_name in ["Data Ingestion", "Data Cleaning"]:
                print("Critical step failed. Stopping pipeline.")
                break
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("     PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    for step_name, step_result in results.items():
        status = step_result['status']
        duration = step_result['duration']
        status_icon = "✓" if status == 'SUCCESS' else "✗"
        print(f"{status_icon} {step_name}: {status} ({duration:.2f}s)")
    
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    run_complete_pipeline()
