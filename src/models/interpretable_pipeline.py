# src/models/interpretable_pipeline.py
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .train_model import ModelTrainingPipeline
from .model_interpreter import ModelInterpreter

class InterpretableModelPipeline(ModelTrainingPipeline):
    """Extended pipeline with model interpretability features"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.interpreters = {}
        
    def create_interpreters(self, X_train: pd.DataFrame):
        """Create model interpreters for all trained models"""
        print(f"\n{'='*60}")
        print("Creating Model Interpreters")
        print('='*60)
        
        for model_type, results in self.results.items():
            print(f"\nCreating interpreter for {model_type.upper()}...")
            
            try:
                # Create interpreter
                interpreter = ModelInterpreter(
                    model=results['model'],
                    model_type=model_type,
                    feature_names=self.data_info['feature_names'],
                    class_names=self.label_encoder.classes_
                )
                
                # Create SHAP explainer
                interpreter.create_explainer(
                    X_train.values,
                    max_samples=self.config.get('shap_max_samples', 100)
                )
                
                self.interpreters[model_type] = interpreter
                print(f"✓ Interpreter for {model_type} created successfully")
                
            except Exception as e:
                print(f"✗ Failed to create interpreter for {model_type}: {str(e)}")
                continue
    
    def generate_interpretability_report(self, X_test: pd.DataFrame, y_test):
        """Generate comprehensive interpretability report"""
        print(f"\n{'='*60}")
        print("Generating Interpretability Report")
        print('='*60)
        
        os.makedirs('reports/interpretability', exist_ok=True)
        
        # Convert y_test to pandas Series if it's a numpy array
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test, index=X_test.index)
        elif not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test, index=X_test.index)
        
        for model_type, interpreter in self.interpreters.items():
            print(f"\n{'-'*40}")
            print(f"Analyzing {model_type.upper()}")
            print('-'*40)
            
            try:
                # Calculate SHAP values for test set
                shap_values, indices = interpreter.explain_predictions(
                    X_test.values,
                    sample_size=min(50, len(X_test))  # Reduced sample size for stability
                )
                
                # Generate plots
                self._generate_shap_plots(model_type, interpreter, X_test, y_test)
                
                # Generate individual explanations
                self._generate_individual_explanations(
                    model_type, interpreter, X_test, y_test, n_samples=3  # Reduced sample size
                )
                
                print(f"✓ Interpretability analysis for {model_type} completed")
                
            except Exception as e:
                print(f"✗ Failed to analyze {model_type}: {str(e)}")
                continue
    
    def _generate_shap_plots(self, model_type: str, interpreter: ModelInterpreter,
                            X_test: pd.DataFrame, y_test: pd.Series):
        """Generate SHAP visualization plots"""
        
        try:
            # Summary plot (bar)
            interpreter.plot_summary(
                plot_type='bar',
                save_path=f'reports/interpretability/{model_type}_shap_summary_bar.png'
            )
            print(f"  ✓ Bar plot saved")
        except Exception as e:
            print(f"  ✗ Bar plot failed: {str(e)}")
        
        try:
            # Summary plot (dot) - often fails with KernelExplainer
            interpreter.plot_summary(
                plot_type='dot',
                save_path=f'reports/interpretability/{model_type}_shap_summary_dot.png'
            )
            print(f"  ✓ Dot plot saved")
        except Exception as e:
            print(f"  ✗ Dot plot failed: {str(e)}")
        
        try:
            # Feature importance
            feature_importance = interpreter.get_feature_importance()
            feature_importance.to_csv(
                f'reports/interpretability/{model_type}_shap_feature_importance.csv',
                index=False
            )
            print(f"  ✓ Feature importance saved")
        except Exception as e:
            print(f"  ✗ Feature importance failed: {str(e)}")
    
    def _generate_individual_explanations(self, model_type: str, 
                                        interpreter: ModelInterpreter,
                                        X_test: pd.DataFrame, 
                                        y_test: pd.Series,
                                        n_samples: int = 3):
        """Generate explanations for individual predictions"""
        
        try:
            # Select samples from different risk levels
            sample_indices = []
            available_classes = y_test.unique()
            
            for risk_level in self.label_encoder.classes_:
                if risk_level in available_classes:
                    # Find instances of this risk level
                    level_mask = y_test == risk_level
                    level_indices = y_test[level_mask].index[:1]  # Just take 1 sample per class
                    sample_indices.extend(level_indices)
            
            # Ensure we don't exceed available samples
            sample_indices = sample_indices[:min(n_samples, len(sample_indices))]
            
            if not sample_indices:
                print(f"  ✗ No valid samples found for explanations")
                return
            
            explanations = []
            
            for idx in sample_indices:
                try:
                    # Get relative position in X_test
                    if idx in X_test.index:
                        relative_idx = X_test.index.get_loc(idx)
                        
                        # Generate explanation
                        explanation = interpreter.explain_student_risk(
                            X_test.iloc[[relative_idx]],
                            student_idx=0
                        )
                        
                        # Add actual risk level
                        explanation['actual_risk'] = y_test.loc[idx]
                        explanation['student_id'] = idx
                        
                        explanations.append(explanation)
                        
                        # Try to generate waterfall plot
                        try:
                            interpreter.plot_waterfall(
                                relative_idx,
                                X_test.values,
                                predicted_class=explanation['prediction'],
                                save_path=f'reports/interpretability/{model_type}_waterfall_student_{idx}.png'
                            )
                        except Exception as e:
                            print(f"    ✗ Waterfall plot for student {idx} failed: {str(e)}")
                            
                except Exception as e:
                    print(f"    ✗ Explanation for student {idx} failed: {str(e)}")
                    continue
            
            if explanations:
                # Save explanations
                self._save_explanations(model_type, explanations)
                print(f"  ✓ {len(explanations)} individual explanations saved")
            else:
                print(f"  ✗ No explanations generated")
                
        except Exception as e:
            print(f"  ✗ Individual explanations failed: {str(e)}")
    
    def _save_explanations(self, model_type: str, explanations: list):
        """Save individual explanations to file"""
        
        try:
            with open(f'reports/interpretability/{model_type}_explanations.txt', 'w') as f:
                f.write(f"Individual Student Risk Explanations - {model_type.upper()}\n")
                f.write("="*80 + "\n\n")
                
                for exp in explanations:
                    f.write(f"Student ID: {exp['student_id']}\n")
                    f.write(f"Actual Risk Level: {exp['actual_risk']}\n")
                    f.write(f"Predicted Risk Level: {exp['predicted_class']}\n")
                    f.write(f"Prediction Confidence: {exp['confidence']:.2%}\n")
                    f.write("\nRisk Probabilities:\n")
                    
                    # Safely handle prediction probabilities
                    try:
                        for i, (class_name, prob) in enumerate(zip(self.label_encoder.classes_, 
                                                                  exp['prediction_proba'])):
                            f.write(f"  {class_name}: {prob:.2%}\n")
                    except Exception:
                        f.write("  [Probabilities unavailable]\n")
                    
                    f.write("\nTop Contributing Factors (Increasing Risk):\n")
                    try:
                        for factor in exp['top_negative_factors']:
                            f.write(f"  - {factor['feature']}: {factor['value']} "
                                   f"(impact: {factor['impact']:.3f})\n")
                    except Exception:
                        f.write("  [Negative factors unavailable]\n")
                    
                    f.write("\nTop Contributing Factors (Decreasing Risk):\n")
                    try:
                        for factor in exp['top_positive_factors']:
                            f.write(f"  - {factor['feature']}: {factor['value']} "
                                   f"(impact: {factor['impact']:.3f})\n")
                    except Exception:
                        f.write("  [Positive factors unavailable]\n")
                    
                    f.write("\nIntervention Recommendations:\n")
                    try:
                        for i, rec in enumerate(exp['intervention_recommendations'], 1):
                            f.write(f"  {i}. {rec}\n")
                    except Exception:
                        f.write("  [Recommendations unavailable]\n")
                    
                    f.write("\n" + "-"*80 + "\n\n")
                    
        except Exception as e:
            print(f"  ✗ Failed to save explanations: {str(e)}")
    
    def run_interpretable_pipeline(self, data_filepath: str, delimiter: str = ','):
        """Run the complete interpretable pipeline"""
        
        try:
            # Run base pipeline
            results = self.run_pipeline(data_filepath, delimiter)
            
            # Load data for interpretability
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data(
                data_filepath, delimiter
            )
            
            # Create interpreters
            self.create_interpreters(X_train)
            
            # Generate interpretability report
            self.generate_interpretability_report(X_test, y_test)
            
            print("\n✓ Interpretability analysis completed!")
            print("Reports saved to: reports/interpretability/")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Interpretable pipeline failed: {str(e)}")
            # Return results from base pipeline if available
            if hasattr(self, 'results'):
                return self.results
            else:
                raise e