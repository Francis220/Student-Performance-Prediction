import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from src.models import ModelFactory

def test_models():
    """Test all three models with synthetic data"""
    
    # Generate synthetic data for testing
    print("Generating synthetic student data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,  # Low, Medium, High, Critical risk levels
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test each model
    models_to_test = ['svm', 'decision_tree', 'knn']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} Model")
        print('='*50)
        
        # Create model
        model = ModelFactory.create_model(model_type)
        
        # Train model
        metrics = model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Store results
        results[model_type] = {
            'model': model,
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Print results
        print(f"\nTraining Accuracy: {metrics['train']['accuracy']:.4f}")
        print(f"Validation Accuracy: {metrics['validation']['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
        
        # Save model
        model.save_model(f'models/{model_type}_model.pkl')
    
    # Compare models
    print(f"\n{'='*50}")
    print("Model Comparison")
    print('='*50)
    print(f"{'Model':<15} {'Train Acc':<10} {'Val Acc':<10} {'CV Acc':<10}")
    print('-'*45)
    
    for model_type, result in results.items():
        metrics = result['metrics']
        print(f"{model_type:<15} "
              f"{metrics['train']['accuracy']:<10.4f} "
              f"{metrics['validation']['accuracy']:<10.4f} "
              f"{metrics['cv_accuracy_mean']:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run tests
    results = test_models()
    print("\nModel testing completed!")