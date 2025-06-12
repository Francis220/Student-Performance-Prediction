from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import BaseModel

class KNNModel(BaseModel):
    """K-Nearest Neighbors model for student risk prediction"""
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        super().__init__('KNN')
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build KNN model"""
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train KNN model with scaling"""
        # Scale the features for KNN
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
        
        # Call parent train method with scaled data
        return super().train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    def predict(self, X):
        """Make predictions with scaling"""
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities with scaling"""
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        
        # Create pipeline with scaler and KNN
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_jobs=-1))
        ])
        
        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Optimizing KNN hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        # Extract the fitted components
        self.scaler = grid_search.best_estimator_.named_steps['scaler']
        self.model = grid_search.best_estimator_.named_steps['knn']
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.best_params
    
    def find_similar_students(self, X, student_index, n_similar=5):
        """Find most similar students to a given student"""
        if not self.is_trained:
            raise Exception("Model must be trained before finding similar students!")
        
        X_scaled = self.scaler.transform(X)
        
        # Get distances and indices of nearest neighbors
        distances, indices = self.model.kneighbors(
            X_scaled[student_index].reshape(1, -1),
            n_neighbors=n_similar + 1
        )
        
        # Remove the student itself from results
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]
        
        return similar_indices, similar_distances
