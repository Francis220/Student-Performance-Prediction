from abc import ABC, abstractmethod
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import json

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = None
        self.training_date = None
        
    @abstractmethod
    def build_model(self):
        """Build the specific model"""
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print(f"Training {self.model_name}...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_date = datetime.now()
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        self.training_metrics['train'] = self.calculate_metrics(y_train, y_pred_train)
        
        # Calculate validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            self.training_metrics['validation'] = self.calculate_metrics(y_val, y_pred_val)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        self.training_metrics['cv_accuracy_mean'] = cv_scores.mean()
        self.training_metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Get feature importance if available
        self.extract_feature_importance(X_train.columns if hasattr(X_train, 'columns') else None)
        
        print(f"{self.model_name} training completed!")
        return self.training_metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise Exception(f"{self.model_name} model is not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise Exception(f"{self.model_name} model is not trained yet!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics
    
    def extract_feature_importance(self, feature_names=None):
        """Extract feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            return None
        
        if feature_names is not None and len(feature_names) == len(importance):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = importance
        
        return self.feature_importance
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance.to_dict() if isinstance(self.feature_importance, pd.DataFrame) else self.feature_importance,
            'training_date': self.training_date.isoformat() if self.training_date else None
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_metrics = model_data['training_metrics']
        
        if isinstance(model_data['feature_importance'], dict):
            self.feature_importance = pd.DataFrame(model_data['feature_importance'])
        else:
            self.feature_importance = model_data['feature_importance']
            
        if model_data['training_date']:
            self.training_date = datetime.fromisoformat(model_data['training_date'])
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'training_metrics': self.training_metrics,
            'model_params': self.model.get_params() if self.model else None
        }
        return info