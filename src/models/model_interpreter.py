import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """Model interpretability using SHAP and other techniques"""
    
    def __init__(self, model, model_type: str, feature_names: List[str], 
                 class_names: List[str] = None):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.shap_values = None
        self.X_sample = None  # Store sample data used for explainer
        
    def create_explainer(self, X_train: np.ndarray, max_samples: int = 100):
        """Create SHAP explainer based on model type"""
        print(f"Creating SHAP explainer for {self.model_type}...")
        
        # Sample data if too large
        if len(X_train) > max_samples:
            sample_indices = np.random.choice(len(X_train), max_samples, replace=False)
            self.X_sample = X_train[sample_indices]
        else:
            self.X_sample = X_train
        
        if self.model_type in ['decision_tree', 'random_forest']:
            if hasattr(self.model, 'named_steps') and 'estimator' in self.model.named_steps:
                estimator = self.model.named_steps['estimator']
            elif hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                estimator = self.model.steps[-1][1]
            else:
                estimator = self.model
            
            self.explainer = shap.TreeExplainer(estimator)
        elif self.model_type == 'gradient_boosting':
            print("Using KernelExplainer for Gradient Boosting (multi-class)")
            def predict_wrapper_gb(X):
                try:
                    return self.model.predict_proba(X)
                except:
                    results = []
                    for i in range(len(X)):
                        try:
                            results.append(self.model.predict_proba(X[i:i+1])[0])
                        except:
                            n_classes = len(self.class_names) if self.class_names else 4
                            results.append(np.ones(n_classes) / n_classes)
                    return np.array(results)
            
            self.explainer = shap.KernelExplainer(
                predict_wrapper_gb,
                self.X_sample,
                link='logit'
            )
        elif self.model_type in ['svm', 'neural_network', 'logistic_regression']:
            def predict_wrapper(X):
                try:
                    return self.model.predict_proba(X)
                except:
                    # Fallback to single predictions for problematic samples
                    results = []
                    for i in range(len(X)):
                        try:
                            results.append(self.model.predict_proba(X[i:i+1])[0])
                        except:
                            # If individual prediction fails, use dummy probabilities
                            n_classes = len(self.class_names) if self.class_names else 4
                            results.append(np.ones(n_classes) / n_classes)
                    return np.array(results)
            
            self.explainer = shap.KernelExplainer(
                predict_wrapper,
                self.X_sample,
                link='logit'
            )
        elif self.model_type == 'knn':
            # Use KernelExplainer for KNN
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.X_sample
            )
        elif self.model_type == 'ensemble':
            # Use KernelExplainer for ensemble models
            def predict_wrapper_ens(X):
                try:
                    return self.model.predict_proba(X)
                except:
                    results = []
                    for i in range(len(X)):
                        try:
                            results.append(self.model.predict_proba(X[i:i+1])[0])
                        except:
                            n_classes = len(self.class_names) if self.class_names else 4
                            results.append(np.ones(n_classes) / n_classes)
                    return np.array(results)
            
            self.explainer = shap.KernelExplainer(
                predict_wrapper_ens,
                self.X_sample
            )
        else:
            print(f"Warning: Unknown model type {self.model_type}, using KernelExplainer")
            # Fallback to KernelExplainer
            def predict_wrapper_default(X):
                try:
                    return self.model.predict_proba(X)
                except:
                    results = []
                    for i in range(len(X)):
                        try:
                            results.append(self.model.predict_proba(X[i:i+1])[0])
                        except:
                            n_classes = len(self.class_names) if self.class_names else 4
                            results.append(np.ones(n_classes) / n_classes)
                    return np.array(results)
            
            self.explainer = shap.KernelExplainer(
                predict_wrapper_default,
                self.X_sample
            )
    
    def explain_predictions(self, X: np.ndarray, sample_size: int = None):
        """Generate SHAP values for predictions"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Sample if needed
        if sample_size and len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            indices = np.arange(len(X))
        
        print(f"Calculating SHAP values for {len(X_sample)} samples...")
        
        # Calculate SHAP values
        if self.model_type in ['svm', 'knn']:
            # KernelExplainer is slow, so limit samples
            n_samples = min(50, len(X_sample))
            self.shap_values = self.explainer.shap_values(X_sample[:n_samples])
        else:
            self.shap_values = self.explainer.shap_values(X_sample)
        
        return self.shap_values, indices
    
    def plot_summary(self, plot_type: str = 'bar', max_display: int = 20, 
                     save_path: str = None):
        """Plot SHAP summary"""
        if self.shap_values is None:
            raise ValueError("No SHAP values calculated. Call explain_predictions first.")
        
        plt.figure(figsize=(10, 8))
        
        # Handle multi-class
        if isinstance(self.shap_values, list):
            # For multi-class, use the first class or aggregate
            shap_values_plot = self.shap_values[0]
        else:
            shap_values_plot = self.shap_values
            
        # Ensure shap_values_plot has the right shape
        if shap_values_plot.shape[1] != len(self.feature_names):
            print(f"Adjusting SHAP values shape from {shap_values_plot.shape} to match features")
            shap_values_plot = shap_values_plot[:, :len(self.feature_names)]
        
        # Create DataFrame for features if needed
        if plot_type == 'dot':
            # For dot plot, we need the actual feature values
            if hasattr(self, 'X_sample') and self.X_sample is not None:
                features = pd.DataFrame(self.X_sample[:shap_values_plot.shape[0], :len(self.feature_names)], 
                                      columns=self.feature_names)
            else:
                features = None
        else:
            features = self.feature_names
        
        # Create summary plot
        shap.summary_plot(
            shap_values_plot,
            features=features,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.title(f'SHAP Feature Importance - {self.model_type.upper()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_waterfall(self, instance_idx: int, X: np.ndarray, 
                      predicted_class: int = None, save_path: str = None):
        """Plot SHAP waterfall for a single prediction"""
        if self.shap_values is None:
            raise ValueError("No SHAP values calculated. Call explain_predictions first.")
        
        # Get SHAP values for the instance
        if isinstance(self.shap_values, list):
            # Multi-class: use values for predicted class
            if predicted_class is None:
                predicted_class = 0
            if instance_idx < len(self.shap_values[predicted_class]):
                instance_shap = self.shap_values[predicted_class][instance_idx]
            else:
                # If instance_idx is out of bounds, use the first instance
                instance_shap = self.shap_values[predicted_class][0]
        else:
            if instance_idx < len(self.shap_values):
                instance_shap = self.shap_values[instance_idx]
            else:
                instance_shap = self.shap_values[0]
        
        # Ensure instance_shap has the right length
        if len(instance_shap) > len(self.feature_names):
            instance_shap = instance_shap[:len(self.feature_names)]
        
        # Get base value
        if isinstance(self.explainer.expected_value, list):
            base_value = self.explainer.expected_value[predicted_class]
        else:
            base_value = self.explainer.expected_value
        
        # Create explanation object
        explanation = shap.Explanation(
            values=instance_shap,
            base_values=base_value,
            data=X[instance_idx][:len(self.feature_names)] if instance_idx < len(X) else X[0][:len(self.feature_names)],
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values"""
        if self.shap_values is None:
            raise ValueError("No SHAP values calculated. Call explain_predictions first.")
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class: average across classes
            importance_list = []
            for sv in self.shap_values:
                if sv.shape[1] > len(self.feature_names):
                    sv = sv[:, :len(self.feature_names)]
                importance_list.append(np.abs(sv).mean(axis=0))
            importance = np.mean(importance_list, axis=0)
        else:
            if self.shap_values.shape[1] > len(self.feature_names):
                shap_vals = self.shap_values[:, :len(self.feature_names)]
            else:
                shap_vals = self.shap_values
            importance = np.abs(shap_vals).mean(axis=0)
        
        # Ensure importance is 1-dimensional
        importance = np.asarray(importance).flatten()
        
        # Ensure we have the right number of features
        if len(importance) != len(self.feature_names):
            # If lengths don't match, truncate or pad as needed
            if len(importance) > len(self.feature_names):
                importance = importance[:len(self.feature_names)]
            else:
                # Pad with zeros if needed
                importance = np.pad(importance, (0, len(self.feature_names) - len(importance)))
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def explain_student_risk(self, student_data: pd.DataFrame, 
                           student_idx: int = 0) -> Dict:
        """Generate comprehensive explanation for a student's risk prediction"""
        # Make prediction
        predictions = self.model.predict(student_data)
        prediction_probas = self.model.predict_proba(student_data)
        
        # Get prediction for the specific student
        prediction = predictions[student_idx]
        prediction_proba = prediction_probas[student_idx]
        
        # Convert prediction to int if needed
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        elif isinstance(prediction, np.ndarray):
            prediction = prediction[0] if len(prediction) > 0 else 0
        prediction = int(prediction)
        
        # Get SHAP values for this student
        shap_values, _ = self.explain_predictions(student_data.values, sample_size=1)
        
        # Get feature contributions
        if isinstance(self.shap_values, list):
            # Multi-class
            if len(self.shap_values) > prediction and len(self.shap_values[prediction]) > 0:
                shap_vals_for_predicted = self.shap_values[prediction][0]
            else:
                shap_vals_for_predicted = self.shap_values[0][0]
                
            # Ensure correct length and flatten
            shap_vals_for_predicted = np.asarray(shap_vals_for_predicted).flatten()
            if len(shap_vals_for_predicted) > len(self.feature_names):
                shap_vals_for_predicted = shap_vals_for_predicted[:len(self.feature_names)]
            elif len(shap_vals_for_predicted) < len(self.feature_names):
                # Pad with zeros if needed
                shap_vals_for_predicted = np.pad(shap_vals_for_predicted, 
                                               (0, len(self.feature_names) - len(shap_vals_for_predicted)))
                
            # Get student values and ensure they're 1-dimensional
            student_values = np.asarray(student_data.iloc[student_idx].values).flatten()
            student_values = student_values[:len(self.feature_names)]
            if len(student_values) < len(self.feature_names):
                student_values = np.pad(student_values, (0, len(self.feature_names) - len(student_values)))
                
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'value': student_values,
                'shap_value': shap_vals_for_predicted,
                'abs_shap': np.abs(shap_vals_for_predicted)
            }).sort_values('abs_shap', ascending=False)
        else:
            if len(self.shap_values) > 0:
                shap_vals = self.shap_values[0]
            else:
                shap_vals = np.zeros(len(self.feature_names))
                
            # Ensure shap_vals is 1-dimensional and correct length
            shap_vals = np.asarray(shap_vals).flatten()
            if len(shap_vals) > len(self.feature_names):
                shap_vals = shap_vals[:len(self.feature_names)]
            elif len(shap_vals) < len(self.feature_names):
                shap_vals = np.pad(shap_vals, (0, len(self.feature_names) - len(shap_vals)))
                
            # Get student values and ensure they're 1-dimensional
            student_values = np.asarray(student_data.iloc[student_idx].values).flatten()
            student_values = student_values[:len(self.feature_names)]
            if len(student_values) < len(self.feature_names):
                student_values = np.pad(student_values, (0, len(self.feature_names) - len(student_values)))
                
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'value': student_values,
                'shap_value': shap_vals,
                'abs_shap': np.abs(shap_vals)
            }).sort_values('abs_shap', ascending=False)
        
        # Get predicted class name
        if self.class_names is not None and len(self.class_names) > prediction:
            predicted_class_name = self.class_names[prediction]
        else:
            predicted_class_name = f"Class_{prediction}"
        
        # Create explanation
        explanation = {
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'predicted_class': predicted_class_name,
            'confidence': prediction_proba.max(),
            'feature_contributions': feature_contributions,
            'top_positive_factors': self._get_top_factors(feature_contributions, positive=True),
            'top_negative_factors': self._get_top_factors(feature_contributions, positive=False),
            'intervention_recommendations': self._generate_recommendations(feature_contributions)
        }
        
        return explanation
    
    def _get_top_factors(self, contributions: pd.DataFrame, positive: bool = True, 
                        top_n: int = 5) -> List[Dict]:
        """Get top contributing factors"""
        if isinstance(contributions, dict):
            # Multi-class: use first class
            contributions = list(contributions.values())[0]
        
        if positive:
            factors = contributions[contributions['shap_value'] > 0].head(top_n)
        else:
            factors = contributions[contributions['shap_value'] < 0].head(top_n)
        
        return [
            {
                'feature': row['feature'],
                'value': row['value'],
                'impact': row['shap_value']
            }
            for _, row in factors.iterrows()
        ]
    
    def _generate_recommendations(self, contributions: pd.DataFrame) -> List[str]:
        """Generate intervention recommendations based on feature contributions"""
        recommendations = []
        
        if isinstance(contributions, dict):
            # Multi-class: use contributions for the predicted high-risk classes
            # Focus on 'Critical' and 'High' risk contributions
            for risk_level in ['Critical', 'High']:
                if risk_level in contributions:
                    contrib = contributions[risk_level]
                    break
            else:
                contrib = list(contributions.values())[0]
        else:
            contrib = contributions
        
        # Analyze top negative contributors
        top_negative = contrib[contrib['shap_value'] < 0].head(5)
        
        for _, row in top_negative.iterrows():
            feature = row['feature']
            value = row['value']
            
            # Generate specific recommendations based on feature
            if 'failures' in feature and value > 0:
                recommendations.append("Provide additional academic support to address past failures")
            elif 'absences' in feature and value > 5:
                recommendations.append("Monitor and address attendance issues")
            elif 'studytime' in feature and value < 2:
                recommendations.append("Encourage increased study time and provide study skills training")
            elif 'schoolsup' in feature and value == 0:
                recommendations.append("Consider enrolling in extra educational support programs")
            elif 'higher' in feature and value == 0:
                recommendations.append("Discuss higher education aspirations and career planning")
            elif 'goout' in feature and value > 3:
                recommendations.append("Balance social activities with academic responsibilities")
            elif 'health' in feature and value < 3:
                recommendations.append("Address health concerns that may impact academic performance")
            elif 'famrel' in feature and value < 3:
                recommendations.append("Consider family counseling to improve home support")
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 3:
            recommendations.extend([
                "Schedule regular check-ins with academic advisor",
                "Connect with peer tutoring services",
                "Develop a personalized study plan"
            ])
        
        return recommendations[:5]  # Return top 5 recommendations