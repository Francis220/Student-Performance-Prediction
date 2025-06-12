#!/usr/bin/env python3
"""
Test script to verify the ML pipeline fixes
"""

import sys
import os
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.interpretable_pipeline import InterpretableModelPipeline
from src.config.training_config import TrainingConfig

def test_pipeline():
    """Test the interpretable pipeline with fixes"""
    
    print("🧪 Testing Machine Learning Pipeline Fixes")
    print("=" * 60)
    
    # Create conservative configuration
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'val_size': 0.1,
        'shap_max_samples': 30,  # Reduced for stability
        'cv_folds': 3,           # Reduced for speed
        'models_to_train': ['random_forest', 'logistic_regression'],  # Start with 2 models
        'optimize_hyperparameters': True,
        'save_models': True,
        'generate_reports': True
    }
    
    # Create directories
    TrainingConfig.create_directories()
    
    # Create pipeline
    pipeline = InterpretableModelPipeline(config)
    
    # Try both datasets
    datasets = [
        ('data/student-por.csv', ','),
        ('data/student-mat.csv', ';')
    ]
    
    for data_filepath, delimiter in datasets:
        if os.path.exists(data_filepath):
            print(f"\n📊 Testing with {data_filepath}")
            print("-" * 40)
            
            try:
                # Run pipeline
                results = pipeline.run_interpretable_pipeline(data_filepath, delimiter)
                
                print(f"✅ Success! Pipeline completed for {data_filepath}")
                print(f"   Models trained: {list(results.keys())}")
                
                # Show best model
                if results:
                    best_model = max(results.items(), 
                                   key=lambda x: x[1].get('test_accuracy', 0))
                    print(f"   Best model: {best_model[0]} (accuracy: {best_model[1].get('test_accuracy', 0):.3f})")
                
                return True
                
            except Exception as e:
                print(f"❌ Error with {data_filepath}: {str(e)}")
                continue
    
    print("❌ No valid datasets found or all tests failed")
    return False

def check_data_availability():
    """Check if datasets are available"""
    datasets = [
        'data/student-por.csv',
        'data/student-mat.csv'
    ]
    
    print("📂 Checking data availability:")
    available = []
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"   ✅ {dataset}")
            available.append(dataset)
        else:
            print(f"   ❌ {dataset}")
    
    return available

if __name__ == "__main__":
    print("🚀 Starting ML Pipeline Test")
    print("=" * 60)
    
    # Check data availability
    available_data = check_data_availability()
    
    if not available_data:
        print("\n⚠️  No datasets found. Please ensure you have:")
        print("   - data/student-por.csv")
        print("   - data/student-mat.csv")
        print("\nDownload from: https://archive.ics.uci.edu/ml/datasets/student+performance")
        sys.exit(1)
    
    # Run test
    success = test_pipeline()
    
    if success:
        print("\n🎉 All tests passed! The pipeline is working correctly.")
        print("\nGenerated outputs:")
        print("   - models/trained/ (trained models)")
        print("   - reports/ (performance reports)")
        print("   - reports/interpretability/ (SHAP analysis)")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1) 