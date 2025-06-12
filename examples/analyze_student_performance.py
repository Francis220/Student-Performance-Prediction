import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from src.data.student_performance_loader import StudentPerformanceDataLoader
from src.models import ModelFactory
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset(df, subject='Portuguese'):
    """Perform exploratory data analysis"""
    print(f"\n{'='*60}")
    print(f"Dataset Analysis - {subject}")
    print('='*60)
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget variable (G3) distribution:")
    print(df['G3'].describe())
    
    # Create risk categories
    loader = StudentPerformanceDataLoader()
    df_risk = loader.create_risk_categories(df, include_g1_g2=False)
    
    print(f"\nRisk level distribution:")
    print(df_risk['risk_level'].value_counts())
    print(f"\nPass/Fail distribution:")
    print(df_risk['pass_fail'].value_counts())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Grade distribution
    axes[0, 0].hist(df['G3'], bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Final Grade (G3)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Final Grades')
    axes[0, 0].axvline(x=10, color='red', linestyle='--', label='Pass threshold')
    axes[0, 0].legend()
    
    # Risk level distribution
    df_risk['risk_level'].value_counts().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Risk Level')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Risk Levels')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Correlation with numeric features
    numeric_features = ['age', 'absences', 'studytime', 'failures', 
                       'famrel', 'freetime', 'goout', 'health']
    correlations = df[numeric_features + ['G3']].corr()['G3'].drop('G3').sort_values()
    
    axes[1, 0].barh(correlations.index, correlations.values)
    axes[1, 0].set_xlabel('Correlation with Final Grade')
    axes[1, 0].set_title('Feature Correlations')
    
    # Gender vs Performance
    gender_performance = df.groupby('sex')['G3'].mean()
    axes[1, 1].bar(gender_performance.index, gender_performance.values)
    axes[1, 1].set_xlabel('Gender')
    axes[1, 1].set_ylabel('Average Final Grade')
    axes[1, 1].set_title('Performance by Gender')
    
    plt.tight_layout()
    plt.savefig(f'analysis_{subject.lower()}.png')
    plt.show()
    
    return df_risk

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Train all models and compare performance"""
    
    # Encode labels for sklearn
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    # Get class names
    class_names = le.classes_
    
    print(f"\n{'='*60}")
    print("Training Machine Learning Models")
    print('='*60)
    
    results = {}
    
    # Train each model
    for model_type in ['svm', 'decision_tree', 'knn']:
        print(f"\n{'-'*40}")
        print(f"Training {model_type.upper()}")
        print('-'*40)
        
        # Create model
        model = ModelFactory.create_model(model_type)
        
        # Train model
        metrics = model.train(
            X_train, y_train_encoded, 
            X_val, y_val_encoded
        )
        
        # Test model
        y_pred = model.predict(X_test)
        test_metrics = model.calculate_metrics(y_test_encoded, y_pred)
        
        # Store results
        results[model_type] = {
            'model': model,
            'train_metrics': metrics,
            'test_metrics': test_metrics,
            'predictions': y_pred
        }
        
        # Print performance
        print(f"\nPerformance Summary:")
        print(f"  Training Accuracy: {metrics['train']['accuracy']:.3f}")
        print(f"  Validation Accuracy: {metrics['validation']['accuracy']:.3f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"  Test Precision: {test_metrics['precision']:.3f}")
        print(f"  Test Recall: {test_metrics['recall']:.3f}")
        print(f"  Test F1-Score: {test_metrics['f1_score']:.3f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save_model(f'models/{model_type}_student_performance.pkl')
        
        # Special handling for decision tree visualization
        if model_type == 'decision_tree':
            try:
                model.visualize_tree(
                    feature_names=feature_names,
                    class_names=class_names,
                    save_path='decision_tree_visualization.png'
                )
                
                # Get tree rules
                rules = model.get_tree_rules(feature_names=feature_names)
                with open('decision_tree_rules.txt', 'w') as f:
                    f.write(rules)
                print("\n  Decision tree visualization and rules saved!")
            except Exception as e:
                print(f"  Could not visualize tree: {e}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            print(f"\n  Top 10 Most Important Features:")
            if isinstance(model.feature_importance, pd.DataFrame):
                for idx, row in model.feature_importance.head(10).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return results, le

def create_comparison_report(results, subject='Portuguese'):
    """Create a comparison report of all models"""
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy comparison
    models = list(results.keys())
    train_acc = [results[m]['train_metrics']['train']['accuracy'] for m in models]
    val_acc = [results[m]['train_metrics']['validation']['accuracy'] for m in models]
    test_acc = [results[m]['test_metrics']['accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[0, 0].bar(x - width, train_acc, width, label='Train')
    axes[0, 0].bar(x, val_acc, width, label='Validation')
    axes[0, 0].bar(x + width, test_acc, width, label='Test')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # Other metrics comparison
    metrics_names = ['precision', 'recall', 'f1_score']
    for idx, metric in enumerate(metrics_names):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        values = [results[m]['test_metrics'][metric] for m in models]
        ax.bar(models, values)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_ylim(0, 1)
        ax.set_xticklabels([m.upper() for m in models])
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{subject.lower()}.png')
    plt.show()
    
    # Print summary report
    print(f"\n{'='*60}")
    print(f"Model Comparison Summary - {subject}")
    print('='*60)
    print(f"\n{'Model':<15} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print('-'*55)
    
    for model_name, result in results.items():
        test_metrics = result['test_metrics']
        print(f"{model_name.upper():<15} "
              f"{test_metrics['accuracy']:<10.3f} "
              f"{test_metrics['precision']:<10.3f} "
              f"{test_metrics['recall']:<10.3f} "
              f"{test_metrics['f1_score']:<10.3f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_metrics']['accuracy'])
    print(f"\nBest performing model: {best_model[0].upper()} "
          f"with test accuracy of {best_model[1]['test_metrics']['accuracy']:.3f}")

def predict_for_sample_students(results, X_test, y_test, le, feature_names, n_samples=5):
    """Show predictions for sample students"""
    print(f"\n{'='*60}")
    print("Sample Student Predictions")
    print('='*60)
    
    # Get best model
    best_model_name = max(results.items(), 
                         key=lambda x: x[1]['test_metrics']['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    # Select random samples
    sample_indices = np.random.choice(X_test.index, n_samples, replace=False)
    
    for idx in sample_indices:
        print(f"\n{'-'*40}")
        print(f"Student Index: {idx}")
        
        # Get student data
        student_data = X_test.loc[[idx]]
        actual_risk = y_test.loc[idx]
        
        # Make prediction
        pred_encoded = best_model.predict(student_data)[0]
        pred_risk = le.inverse_transform([pred_encoded])[0]
        pred_proba = best_model.predict_proba(student_data)[0]
        
        print(f"Actual Risk Level: {actual_risk}")
        print(f"Predicted Risk Level: {pred_risk}")
        print(f"Prediction Confidence: {pred_proba.max():.2%}")
        
        # Show risk probabilities
        print(f"\nRisk Probabilities:")
        for risk_level, prob in zip(le.classes_, pred_proba):
            print(f"  {risk_level}: {prob:.2%}")
        
        # Show key features for this student
        print(f"\nKey Student Features:")
        important_features = ['failures', 'absences', 'studytime', 'higher', 'schoolsup']
        for feat in important_features:
            if feat in student_data.columns:
                value = student_data[feat].values[0]
                print(f"  {feat}: {value}")

def main():
    """Main analysis function"""
    # Load data
    loader = StudentPerformanceDataLoader()
    por_file = 'data/student-por.csv'
    
    if not os.path.exists(por_file):
        print(f"Error: Dataset not found at {por_file}")
        return
    
    # Load and analyze dataset
    df = loader.load_data(por_file, delimiter=',') 
    if df is None:
        return
    
    # Analyze dataset
    df_risk = analyze_dataset(df, subject='Portuguese')
    
    # Prepare features
    X, y = loader.prepare_features(df_risk, target_col='risk_level')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Train and evaluate models
    results, le = train_and_evaluate_models(
        X_train, X_val, X_test, 
        y_train, y_val, y_test, 
        feature_names
    )
    
    # Create comparison report
    create_comparison_report(results, subject='Portuguese')
    
    # Show sample predictions
    predict_for_sample_students(results, X_test, y_test, le, feature_names)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - analysis_portuguese.png: Data analysis visualizations")
    print("  - model_comparison_portuguese.png: Model performance comparison")
    print("  - decision_tree_visualization.png: Decision tree structure")
    print("  - decision_tree_rules.txt: Decision tree rules in text format")
    print("  - models/: Saved trained models")

if __name__ == "__main__":
    main()