import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.interpretable_pipeline import InterpretableModelPipeline

def main():
    
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'val_size': 0.1,
        'shap_max_samples': 100,
        'cv_folds': 5
    }
    
    pipeline = InterpretableModelPipeline(config)
    
    data_filepath = 'data/student-mat.csv'
    
    print("🚀 Starting Advanced Student Performance Prediction Pipeline")
    print("=" * 70)
    
    try:
        results = pipeline.run_interpretable_pipeline(data_filepath, delimiter=';')
        
        print("\n" + "=" * 70)
        print("📊 ADVANCED PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        
        print("\n🤖 Models Trained:")
        for model_name in results.keys():
            accuracy = results[model_name].get('test_accuracy', 'N/A')
            if isinstance(accuracy, float):
                print(f"  - {model_name.replace('_', ' ').title()}: {accuracy:.3f}")
            else:
                print(f"  - {model_name.replace('_', ' ').title()}: {accuracy}")
        
        print("\n📁 Generated Outputs:")
        print("  - Enhanced trained models: models/trained/")
        print("  - Advanced performance reports: reports/")
        print("  - Deep interpretability analysis: reports/interpretability/")
        print("  - Feature importance with interactions")
        print("  - Individual student risk explanations")
        print("  - Advanced SHAP visualizations")
        
        best_model_name = max(results.keys(), 
                            key=lambda k: results[k].get('test_accuracy', 0))
        best_accuracy = results[best_model_name].get('test_accuracy', 0)
        
        print(f"\n🏆 Best Performing Model:")
        print(f"  {best_model_name.replace('_', ' ').title()} with test accuracy: {best_accuracy:.3f}")
        
        if best_accuracy > 0.5:
            print("  🎉 Excellent performance achieved!")
        elif best_accuracy > 0.45:
            print("  ✅ Good performance achieved!")
        else:
            print("  ⚠️  Consider additional feature engineering")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 