#!/usr/bin/env python3
"""
Flask Web Application for Student Performance Prediction
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.utils
import pickle
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.interpretable_pipeline import InterpretableModelPipeline
from src.config.training_config import TrainingConfig

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for loaded models and data
loaded_models = {}
current_data = None
model_performance = {}
feature_importance_data = {}

class WebAppInterface:
    """Interface class for web application functionality"""
    
    def __init__(self):
        self.pipeline = None
        self.models = {}
        self.data_info = {}
        
    def load_trained_models(self):
        """Load all trained models"""
        models_dir = 'models/trained'
        if not os.path.exists(models_dir):
            return False
            
        try:
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):
                    model_name = filename.replace('.pkl', '')
                    model_path = os.path.join(models_dir, filename)
                    
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        
            return len(self.models) > 0
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def load_model_performance(self):
        """Load model performance metrics"""
        reports_dir = 'reports'
        performance = {}
        
        try:
            # Look for performance reports
            for filename in os.listdir(reports_dir):
                if filename.endswith('_performance.json'):
                    model_name = filename.replace('_performance.json', '')
                    
                    with open(os.path.join(reports_dir, filename), 'r') as f:
                        performance[model_name] = json.load(f)
                        
        except Exception as e:
            print(f"Error loading performance data: {e}")
            
        return performance
    
    def get_feature_importance(self):
        """Load feature importance data"""
        interpretability_dir = 'reports/interpretability'
        importance_data = {}
        
        if not os.path.exists(interpretability_dir):
            return importance_data
            
        try:
            for filename in os.listdir(interpretability_dir):
                if filename.endswith('_shap_feature_importance.csv'):
                    model_name = filename.replace('_shap_feature_importance.csv', '')
                    
                    df = pd.read_csv(os.path.join(interpretability_dir, filename))
                    importance_data[model_name] = df.to_dict('records')
                    
        except Exception as e:
            print(f"Error loading feature importance: {e}")
            
        return importance_data

# Initialize the web app interface
web_interface = WebAppInterface()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    # Load models and performance data
    models_loaded = web_interface.load_trained_models()
    performance_data = web_interface.load_model_performance()
    importance_data = web_interface.get_feature_importance()
    
    # Get available datasets
    available_datasets = []
    data_dir = 'data'
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                available_datasets.append(file)
    
    # Create model comparison chart
    comparison_chart = create_model_comparison_chart(performance_data)
    
    return render_template('dashboard.html',
                         models_loaded=models_loaded,
                         models=list(web_interface.models.keys()),
                         performance_data=performance_data,
                         importance_data=importance_data,
                         available_datasets=available_datasets,
                         comparison_chart=comparison_chart)

@app.route('/predict')
def predict_page():
    """Student prediction page"""
    models_available = len(web_interface.models) > 0
    return render_template('predict.html', models_available=models_available)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Load models if not already loaded
        if not web_interface.models:
            success = web_interface.load_trained_models()
            if not success:
                return jsonify({'error': 'No trained models available'}), 404
        
        # Extract features from the request
        features = extract_features_from_request(data)
        
        # Make predictions with all available models
        predictions = {}
        
        for model_name, model in web_interface.models.items():
            try:
                # Make prediction
                prediction = model.predict([features])[0]
                probability = model.predict_proba([features])[0] if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = {
                    'prediction': prediction,
                    'probability': probability.tolist() if probability is not None else None
                }
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        
        return jsonify({
            'predictions': predictions,
            'input_features': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-insights')
def model_insights():
    """Model insights and interpretability page"""
    importance_data = web_interface.get_feature_importance()
    
    # Get available SHAP plots
    shap_plots = get_available_shap_plots()
    
    return render_template('model_insights.html',
                         importance_data=importance_data,
                         shap_plots=shap_plots)

@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    """Model training interface"""
    if request.method == 'POST':
        try:
            # Get training configuration from form
            config = {
                'random_state': int(request.form.get('random_state', 42)),
                'test_size': float(request.form.get('test_size', 0.2)),
                'val_size': float(request.form.get('val_size', 0.1)),
                'cv_folds': int(request.form.get('cv_folds', 5)),
                'models_to_train': request.form.getlist('models_to_train'),
                'optimize_hyperparameters': 'optimize_hyperparameters' in request.form,
                'save_models': True,
                'generate_reports': True
            }
            
            # Get dataset
            dataset = request.form.get('dataset')
            if not dataset:
                flash('Please select a dataset', 'error')
                return redirect(url_for('train_model'))
            
            # Determine delimiter
            delimiter = ';' if 'mat' in dataset else ','
            data_path = os.path.join('data', dataset)
            
            if not os.path.exists(data_path):
                flash('Dataset not found', 'error')
                return redirect(url_for('train_model'))
            
            # Start training process
            pipeline = InterpretableModelPipeline(config)
            results = pipeline.run_interpretable_pipeline(data_path, delimiter)
            
            if results:
                flash('Model training completed successfully!', 'success')
                # Reload models
                web_interface.load_trained_models()
            else:
                flash('Model training failed', 'error')
                
        except Exception as e:
            flash(f'Training error: {str(e)}', 'error')
        
        return redirect(url_for('train_model'))
    
    # GET request - show training form
    available_datasets = []
    data_dir = 'data'
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                available_datasets.append(file)
    
    available_models = ['svm', 'decision_tree', 'knn', 'random_forest', 
                       'logistic_regression', 'neural_network']
    
    return render_template('train_model.html',
                         available_datasets=available_datasets,
                         available_models=available_models)

@app.route('/api/feature-importance/<model_name>')
def api_feature_importance(model_name):
    """API endpoint for feature importance data"""
    importance_data = web_interface.get_feature_importance()
    
    if model_name in importance_data:
        return jsonify(importance_data[model_name])
    else:
        return jsonify({'error': 'Feature importance data not found'}), 404

@app.route('/api/model-performance')
def api_model_performance():
    """API endpoint for model performance data"""
    performance_data = web_interface.load_model_performance()
    return jsonify(performance_data)

def extract_features_from_request(data):
    """Extract and process features from API request"""
    # Define expected features based on the student dataset
    feature_mapping = {
        'age': int,
        'medu': int,  # Mother's education
        'fedu': int,  # Father's education
        'traveltime': int,
        'studytime': int,
        'failures': int,
        'famrel': int,  # Family relationship quality
        'freetime': int,
        'goout': int,
        'dalc': int,  # Workday alcohol consumption
        'walc': int,  # Weekend alcohol consumption
        'health': int,
        'absences': int,
        'school_gp': lambda x: 1 if x == 'GP' else 0,
        'sex_m': lambda x: 1 if x == 'M' else 0,
        'address_u': lambda x: 1 if x == 'U' else 0,
        'famsize_le3': lambda x: 1 if x == 'LE3' else 0,
        'pstatus_t': lambda x: 1 if x == 'T' else 0,
        'mjob_services': lambda x: 1 if x == 'services' else 0,
        'mjob_other': lambda x: 1 if x == 'other' else 0,
        'mjob_health': lambda x: 1 if x == 'health' else 0,
        'mjob_teacher': lambda x: 1 if x == 'teacher' else 0,
        'fjob_services': lambda x: 1 if x == 'services' else 0,
        'fjob_other': lambda x: 1 if x == 'other' else 0,
        'fjob_health': lambda x: 1 if x == 'health' else 0,
        'fjob_teacher': lambda x: 1 if x == 'teacher' else 0,
        'reason_course': lambda x: 1 if x == 'course' else 0,
        'reason_other': lambda x: 1 if x == 'other' else 0,
        'reason_home': lambda x: 1 if x == 'home' else 0,
        'guardian_mother': lambda x: 1 if x == 'mother' else 0,
        'guardian_father': lambda x: 1 if x == 'father' else 0,
        'schoolsup': lambda x: 1 if x == 'yes' else 0,
        'famsup': lambda x: 1 if x == 'yes' else 0,
        'paid': lambda x: 1 if x == 'yes' else 0,
        'activities': lambda x: 1 if x == 'yes' else 0,
        'nursery': lambda x: 1 if x == 'yes' else 0,
        'higher': lambda x: 1 if x == 'yes' else 0,
        'internet': lambda x: 1 if x == 'yes' else 0,
        'romantic': lambda x: 1 if x == 'yes' else 0
    }
    
    features = []
    for feature, converter in feature_mapping.items():
        value = data.get(feature, 0)
        try:
            features.append(converter(value))
        except:
            features.append(0)  # Default value for missing/invalid data
    
    return features

def create_model_comparison_chart(performance_data):
    """Create a plotly chart comparing model performance"""
    if not performance_data:
        return None
    
    models = list(performance_data.keys())
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model in models:
        perf = performance_data[model]
        accuracies.append(perf.get('test_accuracy', 0) * 100)
        precisions.append(perf.get('test_precision', 0) * 100)
        recalls.append(perf.get('test_recall', 0) * 100)
        f1_scores.append(perf.get('test_f1', 0) * 100)
    
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracies),
        go.Bar(name='Precision', x=models, y=precisions),
        go.Bar(name='Recall', x=models, y=recalls),
        go.Bar(name='F1-Score', x=models, y=f1_scores)
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score (%)',
        barmode='group',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_available_shap_plots():
    """Get list of available SHAP plot files"""
    plots = {}
    interpretability_dir = 'reports/interpretability'
    
    if not os.path.exists(interpretability_dir):
        return plots
    
    for filename in os.listdir(interpretability_dir):
        if filename.endswith('.png'):
            # Parse filename to get model and plot type
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 2:
                model_name = parts[0]
                plot_type = '_'.join(parts[1:])
                
                if model_name not in plots:
                    plots[model_name] = []
                
                plots[model_name].append({
                    'filename': filename,
                    'plot_type': plot_type,
                    'path': f'/{interpretability_dir}/{filename}'
                })
    
    return plots

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
