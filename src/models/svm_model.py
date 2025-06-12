from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from .base_model import BaseModel

class SVMModel(BaseModel):
    """Support Vector Machine model for student risk prediction"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True):
        super().__init__('SVM')
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        
    def build_model(self):
        """Build SVM model"""
        # Use SVC with probability=True for probability predictions
        base_svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        # Wrap in OneVsRest for multi-class classification
        self.model = OneVsRestClassifier(base_svm, n_jobs=-1)
        
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'estimator__kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        base_svm = SVC(probability=True, random_state=42, class_weight='balanced')
        model = OneVsRestClassifier(base_svm)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Optimizing SVM hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.best_params
