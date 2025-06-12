import os

class TrainingConfig:
    """Configuration for model training pipeline"""
    
    # Data configuration
    DATA_CONFIG = {
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42,
        'include_g1_g2': False  # For early prediction
    }
    
    # Model configuration
    MODEL_CONFIG = {
        'models_to_train': ['svm', 'decision_tree', 'knn'],
        'optimize_hyperparameters': True,
        'cv_folds': 5
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        'batch_size': 32,
        'early_stopping': True,
        'patience': 10,
        'save_best_only': True
    }
    
    # Output configuration
    OUTPUT_CONFIG = {
        'save_models': True,
        'generate_reports': True,
        'model_dir': 'models/trained',
        'report_dir': 'reports',
        'log_dir': 'logs'
    }
    
    # SHAP configuration (for Step 4)
    SHAP_CONFIG = {
        'max_samples': 100,
        'plot_type': 'bar',
        'save_plots': True
    }
    
    @classmethod
    def get_full_config(cls):
        """Get complete configuration"""
        return {
            **cls.DATA_CONFIG,
            **cls.MODEL_CONFIG,
            **cls.TRAINING_CONFIG,
            **cls.OUTPUT_CONFIG,
            **cls.SHAP_CONFIG
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.OUTPUT_CONFIG['model_dir'],
            cls.OUTPUT_CONFIG['report_dir'],
            cls.OUTPUT_CONFIG['log_dir']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)