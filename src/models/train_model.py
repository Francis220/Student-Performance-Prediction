import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import warnings

# Suppress all warnings to reduce noise
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress numpy warnings
np.seterr(all='ignore')

from src.data.preprocessing import DataPreprocessor
from .model_factory import ModelFactory

class ModelTrainingPipeline:
    """Complete pipeline for training and evaluating models"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.results = {}
        self.trained_models_dir = 'models/trained'
        os.makedirs(self.trained_models_dir, exist_ok=True)
        self.label_encoder = None
        self.data_info = {}
        
    def _default_config(self):
        """Default configuration for training"""
        return {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'cv_folds': 5,
            'include_g1_g2': False,
            'models_to_train': ['svm', 'decision_tree', 'knn'],
            'optimize_hyperparameters': True,
            'save_models': True,
            'generate_reports': True
        }
    
    def load_and_prepare_data(self, filepath, delimiter=','):
        """Load and prepare data for training"""
        print("Loading and preparing data...")
        df = pd.read_csv(filepath, delimiter=delimiter)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        print("Excluding G1 and G2 from features (more realistic early prediction)")
        exclude_cols = ['G1', 'G2'] if 'G1' in df.columns else []
        
        X, y = self.preprocessor.preprocess_data(filepath, delimiter, exclude_cols=exclude_cols)
        
        y_categories = self.preprocessor.create_risk_categories(y, method='quantile')
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_categories)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
        )
        
        self.data_info = {
            'feature_names': X.columns.tolist(),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_
        }
        
        print(f"Data split completed:")
        print(f"  Train set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train all specified models"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Store the test labels for later use
        self.y_test_encoded = y_test_encoded  # Add this line
        
        print(f"\n{'='*60}")
        print("Training Models")
        print('='*60)
        
        for model_type in self.config['models_to_train']:
            print(f"\n{'-'*40}")
            print(f"Training {model_type.upper()}")
            print('-'*40)
            
            # Create model
            model = ModelFactory.create_model(model_type)
            
            # Optimize hyperparameters if requested
            if self.config['optimize_hyperparameters']:
                print("Optimizing hyperparameters...")
                model.optimize_hyperparameters(X_train, y_train_encoded)
            
            # Train model
            train_start = datetime.now()
            metrics = model.train(X_train, y_train_encoded, X_val, y_val_encoded)
            train_time = (datetime.now() - train_start).total_seconds()
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            test_metrics = model.calculate_metrics(y_test_encoded, y_pred)
            
            # Perform cross-validation
            cv_results = self._perform_cross_validation(
                model, X_train, y_train_encoded
            )
            
            # Store results
            self.models[model_type] = model
            self.results[model_type] = {
                'model': model,
                'train_metrics': metrics,
                'test_metrics': test_metrics,
                'cv_results': cv_results,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'training_time': train_time,
                'feature_importance': model.feature_importance,
                'y_true': y_test_encoded,  # Add this line to store true labels
                'y_test': y_test  # Add this line to store original labels
            }
            self._print_model_summary(model_type)
            if self.config['save_models']:
                self._save_model(model_type, model)
    
    def _perform_cross_validation(self, model, X, y):
        """Perform cross-validation"""
        skf = StratifiedKFold(
            n_splits=self.config['cv_folds'], 
            shuffle=True, 
            random_state=self.config['random_state']
        )
        
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = cross_validate(
            model.model, X, y, 
            cv=skf, 
            scoring=scoring,
            return_train_score=True
        )
        
        return {
            'test_accuracy': cv_results['test_accuracy'].mean(),
            'test_accuracy_std': cv_results['test_accuracy'].std(),
            'test_precision': cv_results['test_precision_weighted'].mean(),
            'test_recall': cv_results['test_recall_weighted'].mean(),
            'test_f1': cv_results['test_f1_weighted'].mean()
        }
    
    def _print_model_summary(self, model_type):
        """Print model performance summary"""
        results = self.results[model_type]
        
        print(f"\nPerformance Summary for {model_type.upper()}:")
        print(f"  Training Accuracy: {results['train_metrics']['train']['accuracy']:.3f}")
        print(f"  Validation Accuracy: {results['train_metrics']['validation']['accuracy']:.3f}")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.3f}")
        print(f"  CV Accuracy: {results['cv_results']['test_accuracy']:.3f} ± {results['cv_results']['test_accuracy_std']:.3f}")
        print(f"  Training Time: {results['training_time']:.2f} seconds")
    
    def _save_model(self, model_type, model):
        """Save trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_model_{timestamp}.pkl"
        filepath = os.path.join(self.trained_models_dir, filename)
        
        model.save_model(filepath)
        metadata = {
            'model_type': model_type,
            'training_date': timestamp,
            'config': self.config,
            'data_info': self.data_info,
            'performance': {
                'test_accuracy': self.results[model_type]['test_metrics']['accuracy'],
                'cv_accuracy': self.results[model_type]['cv_results']['test_accuracy']
            }
        }
        
        metadata_filepath = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if self.config['generate_reports']:
            print(f"\n{'='*60}")
            print("Generating Comparison Report")
            print('='*60)
            
            self._plot_model_comparison()
            self._plot_roc_curves()
            self._plot_feature_importance_comparison()
            self._generate_text_report()
    
    def _plot_model_comparison(self):
        """Plot model comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.results.keys())
        metrics = ['train', 'validation', 'test', 'cv']
        accuracies = {
            'train': [self.results[m].get('train_accuracy', 0) for m in models],
            'validation': [self.results[m].get('val_accuracy', 0) for m in models], 
            'test': [self.results[m].get('test_accuracy', 0) for m in models],
            'cv': [self.results[m].get('cv_mean', 0) for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i*width, accuracies[metric], width, label=metric.capitalize())
        
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels([m.upper() for m in models])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].text(0.5, 0.5, 'Additional metrics\navailable in\ndetailed report', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Detailed Metrics')
        
        axes[1, 0].text(0.5, 0.5, 'Feature importance\navailable in\nseparate plots', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
        
        axes[1, 1].text(0.5, 0.5, 'Cross-validation\nscores included\nin accuracy plot', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('CV Performance')
        
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            try:
                y_score = results['probabilities']
                y_true = results['y_true']
                
                y_test_bin = label_binarize(
                    y_true,
                    classes=np.arange(len(self.label_encoder.classes_))
                )
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(len(self.label_encoder.classes_)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    y_test_bin.ravel(), y_score.ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                plt.plot(
                    fpr["micro"], tpr["micro"],
                    label=f'{model_name.upper()} (AUC = {roc_auc["micro"]:.3f})',
                    linewidth=2
                )
            except Exception as e:
                print(f"Could not plot ROC curve for {model_name}: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        os.makedirs('reports', exist_ok=True) 
        plt.savefig('reports/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance_comparison(self):
        """Plot feature importance comparison"""
        try:
            # Create placeholder plot since feature importance handling varies by model
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'Feature Importance Analysis\n\nDetailed feature importance available in:\n'
                     '- Model-specific plots\n- Interpretability reports\n- SHAP analysis\n'
                     '- Individual model outputs', 
                     ha='center', va='center', fontsize=14)
            plt.title('Feature Importance Summary', fontsize=16)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('reports/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not generate feature importance plot: {e}")
    
    def _generate_text_report(self):
        """Generate detailed text report"""
        os.makedirs('reports', exist_ok=True)
        
        with open('reports/model_comparison_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.data_info.items():
                if key == 'feature_names':
                    f.write(f"{key}: {len(value)} features\n")
                elif key == 'class_distribution':
                    f.write(f"{key}:\n")
                    for class_name, count in value.items():
                        f.write(f"  {class_name}: {count}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Model results
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*20 + "\n")
                
                # Handle different result structures
                if 'test_score' in results:
                    f.write(f"Test Accuracy: {results['test_score']:.4f}\n")
                elif 'test_metrics' in results:
                    test_metrics = results['test_metrics']
                    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
                    f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
                    f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
                    f.write(f"Test F1-Score: {test_metrics['f1_score']:.4f}\n")
                
                if 'cv_scores' in results:
                    cv_scores = results['cv_scores']
                    f.write(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
                elif 'cv_results' in results:
                    cv_results = results['cv_results']
                    f.write(f"CV Accuracy: {cv_results['test_accuracy']:.4f} ± {cv_results['test_accuracy_std']:.4f}\n")
                
                f.write(f"Training Time: {results.get('training_time', 0):.2f} seconds\n")
                
                # Classification report if available
                if 'y_true' in results and 'predictions' in results:
                    f.write("\nClassification Report:\n")
                    f.write(classification_report(
                        results['y_true'],
                        results['predictions'],
                        target_names=self.label_encoder.classes_
                    ))
            
            # Best model
            def get_accuracy(results):
                if 'test_score' in results:
                    return results['test_score']
                elif 'test_metrics' in results:
                    return results['test_metrics']['accuracy']
                else:
                    return 0
            
            best_model = max(self.results.items(), key=lambda x: get_accuracy(x[1]))
            f.write(f"\n{'='*40}\n")
            f.write(f"BEST MODEL: {best_model[0].upper()}\n")
            f.write(f"Test Accuracy: {get_accuracy(best_model[1]):.4f}\n")
    
    def create_advanced_models(self):
        """Create advanced models for training"""
        models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'svm': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', SVC(probability=True, random_state=42))
                ]),
                'param_grid': {
                    'estimator__C': [0.1, 1, 10],
                    'estimator__kernel': ['rbf', 'poly', 'linear'],
                    'estimator__gamma': ['scale', 'auto', 0.01, 0.1]
                }
            },
            'neural_network': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', MLPClassifier(
                        random_state=42, 
                        max_iter=2000,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        tol=1e-6,
                        solver='adam',
                        batch_size='auto'
                    ))
                ]),
                'param_grid': {
                    'estimator__hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'estimator__activation': ['relu', 'tanh'],
                    'estimator__alpha': [0.0001, 0.001, 0.01],
                    'estimator__learning_rate_init': [0.001, 0.01]
                }
            },
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', LogisticRegression(
                        random_state=42, 
                        max_iter=2000,
                        tol=1e-6,
                        solver='saga',
                        n_jobs=-1
                    ))
                ]),
                'param_grid': {
                    'estimator__C': [0.01, 0.1, 1, 10],
                    'estimator__penalty': ['l1', 'l2', 'elasticnet'],
                    'estimator__solver': ['saga'],
                    'estimator__class_weight': ['balanced', None],
                    'estimator__l1_ratio': [0.1, 0.5, 0.9]
                }
            }
        }
        
        return models
    
    def optimize_hyperparameters(self, model, param_grid, X_train, y_train, model_name):
        """Optimize hyperparameters for a given model"""
        print(f"Optimizing {model_name} hyperparameters...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble_model(self, trained_models):
        """Create an ensemble model from trained models"""
        ensemble_models = []
        
        for name, result in trained_models.items():
            if name in ['random_forest', 'gradient_boosting', 'svm']:
                ensemble_models.append((name, result['model']))
        
        if len(ensemble_models) >= 3:
            voting_clf = VotingClassifier(
                estimators=ensemble_models, 
                voting='soft'
            )
            return voting_clf
        
        return None
    
    def train_single_model(self, model_config, X_train, X_val, X_test, 
                          y_train, y_val, y_test, model_name):
        """Train a single model and store the results"""
        print(f"\n{'-'*40}")
        print(f"Training {model_name.upper().replace('_', ' ')}")
        print('-'*40)
        
        print("Optimizing hyperparameters...")
        start_time = datetime.now()
        
        best_model = self.optimize_hyperparameters(
            model_config['model'], 
            model_config['param_grid'],
            X_train, y_train, 
            model_name
        )
        
        print(f"Training {model_name}...")
        best_model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        test_pred = best_model.predict(X_test)
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        
        results = {
            'model': best_model,
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
        }
        
        print(f"{model_name} training completed!")
        print(f"\nPerformance Summary for {model_name.upper().replace('_', ' ')}:")
        print(f"  Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"  Validation Accuracy: {results['val_accuracy']:.3f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"  CV Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"  Training Time: {results['training_time']:.2f} seconds")
        
        model_filename = f"{model_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(self.trained_models_dir, model_filename)
        os.makedirs(self.trained_models_dir, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved to {model_path}")
        
        return results
    
    def run_pipeline(self, data_filepath, delimiter=','):
        """Run the complete training pipeline"""
        print("Starting Advanced Model Training Pipeline...")
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data(
            data_filepath, delimiter
        )
        
        print(f"\n{'='*60}")
        print("Training Advanced Models")
        print('='*60)
        
        models = self.create_advanced_models()
        
        for model_name, model_config in models.items():
            self.results[model_name] = self.train_single_model(
                model_config, X_train, X_val, X_test, 
                y_train, y_val, y_test, model_name
            )
        
        print(f"\n{'='*60}")
        print("Creating Ensemble Model")
        print('='*60)
        
        ensemble_model = self.create_ensemble_model(self.results)
        if ensemble_model:
            print("Training ensemble model...")
            ensemble_model.fit(X_train, y_train)
            
            ensemble_test_pred = ensemble_model.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_test_pred)
            
            self.results['ensemble'] = {
                'model': ensemble_model,
                'test_accuracy': ensemble_accuracy,
                'predictions': {'test': ensemble_test_pred}
            }
            
            print(f"Ensemble Test Accuracy: {ensemble_accuracy:.3f}")
        
        self.generate_comparison_report()
        
        print("Advanced pipeline completed successfully!")
        print("Models saved to: models/trained")
        print("Reports saved to: reports/")
        
        return self.results