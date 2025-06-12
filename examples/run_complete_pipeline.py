import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.interpretable_pipeline import InterpretableModelPipeline
from src.config.training_config import TrainingConfig

def main():
    """Run the complete interpretable model pipeline"""
    
    # Create directories
    TrainingConfig.create_directories()
    
    # Get configuration
    config = TrainingConfig.get_full_config()
    
    # Create pipeline
    pipeline = InterpretableModelPipeline(config)
    data_filepath = 'data/student-por.csv'
    
    # Check if file exists
    if not os.path.exists(data_filepath):
        print(f"Error: Dataset not found at {data_filepath}")
        print("Please ensure the studentpor.csv file is in the correct location.")
        return
    
    # Run complete pipeline with interpretability
    results = pipeline.run_interpretable_pipeline(data_filepath, delimiter=',')
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    print("\nModels Trained:")
    for model_type in results.keys():
        print(f"  - {model_type.upper()}")
    
    print("\nGenerated Outputs:")
    print("  - Trained models: models/trained/")
    print("  - Performance reports: reports/")
    print("  - Interpretability analysis: reports/interpretability/")
    print("  - Feature importance rankings")
    print("  - Individual student explanations")
    print("  - SHAP visualizations")
    
    if results:
        print("\nBest Performing Model:")
        def get_test_accuracy(result_item):
            model_name, result_data = result_item
            if 'test_accuracy' in result_data:
                return result_data['test_accuracy']
            elif 'test_metrics' in result_data and 'accuracy' in result_data['test_metrics']:
                return result_data['test_metrics']['accuracy']
            else:
                return 0.0
        
        best_model = max(results.items(), key=get_test_accuracy)
        best_accuracy = get_test_accuracy(best_model)
        print(f"  {best_model[0].upper()} with test accuracy: {best_accuracy:.3f}")
    else:
        print("\nNo models were successfully trained.")
        print("Check the error messages above for troubleshooting information.")

if __name__ == "__main__":
    main()