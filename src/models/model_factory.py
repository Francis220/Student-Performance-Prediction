from .svm_model import SVMModel
from .decision_tree_model import DecisionTreeModel
from .knn_model import KNNModel

class ModelFactory:
    """Factory class to create different ML models"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a model based on the specified type
        
        Parameters:
        -----------
        model_type : str
            Type of model ('svm', 'decision_tree', 'knn')
        **kwargs : dict
            Model-specific parameters
            
        Returns:
        --------
        BaseModel instance
        """
        model_type = model_type.lower()
        
        if model_type == 'svm':
            return SVMModel(**kwargs)
        elif model_type == 'decision_tree' or model_type == 'dt':
            return DecisionTreeModel(**kwargs)
        elif model_type == 'knn':
            return KNNModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Get list of available model types"""
        return ['svm', 'decision_tree', 'knn']
    
    @staticmethod
    def get_model_params(model_type):
        """Get default parameters for a model type"""
        params = {
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True
            },
            'decision_tree': {
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10
            },
            'knn': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'metric': 'euclidean'
            }
        }
        return params.get(model_type.lower(), {})