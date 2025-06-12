from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree model for student risk prediction"""
    
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10):
        super().__init__('Decision Tree')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def build_model(self):
        """Build Decision Tree model"""
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            class_weight='balanced'
        )
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [10, 20, 50, 100],
            'min_samples_leaf': [5, 10, 20, 50],
            'criterion': ['gini', 'entropy']
        }
        
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Optimizing Decision Tree hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.best_params
    
    def visualize_tree(self, feature_names=None, class_names=None, save_path=None):
        """Visualize the decision tree"""
        if not self.is_trained:
            raise Exception("Model must be trained before visualization!")
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to {save_path}")
        else:
            plt.show()
    
    def get_tree_rules(self, feature_names=None, class_names=None):
        """Extract tree rules as text"""
        if not self.is_trained:
            raise Exception("Model must be trained before extracting rules!")
        
        tree_rules = export_text(
            self.model,
            feature_names=list(feature_names) if feature_names is not None else None
        )
        return tree_rules
