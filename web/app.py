#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

@app.route('/')
def dashboard():
    total_students = 0
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    high_risk_students = []
    medium_risk_students = []
    low_risk_students = []
    
    student_por_path = 'data/student-por.csv'
    if os.path.exists(student_por_path):
        student_por_df = pd.read_csv(student_por_path)
        student_por_df['G3'] = pd.to_numeric(student_por_df['G3'], errors='coerce')
        total_students = len(student_por_df)
        high_risk_count = len(student_por_df[student_por_df['G3'] <= 9])
        medium_risk_count = len(student_por_df[(student_por_df['G3'] >= 10) & (student_por_df['G3'] <= 14)])
        low_risk_count = len(student_por_df[student_por_df['G3'] >= 15])

        risk_students = {'high': [], 'medium': [], 'low': []}

        for idx, row in student_por_df.iterrows():
            g3 = row['G3']
            if pd.isna(g3):
                continue

            if g3 <= 9:
                level = 'high'
            elif g3 <= 14:
                level = 'medium'
            else:
                level = 'low'

            factors = []
            try:
                if row.get('failures', 0) > 0:
                    factors.append('Past Failures')
                if row.get('absences', 0) > 5:
                    factors.append('Frequent Absences')
                if row.get('studytime', 2) <= 1:
                    factors.append('Low Study Time')
                if row.get('goout', 0) >= 4:
                    factors.append('High Social Activity')
                if row.get('Dalc', 0) >= 3 or row.get('Walc', 0) >= 3:
                    factors.append('Alcohol Consumption')
                if row.get('schoolsup', 'yes') == 'no':
                    factors.append('No School Support')
                if row.get('famsup', 'yes') == 'no':
                    factors.append('No Family Support')
            except Exception:
                pass

            if level == 'low' and not factors:
                factors.append('Strong Performance')

            risk_students[level].append({
                'id': f"Student #{idx + 1}",
                'factors': factors[:3]
            })

        high_risk_students = risk_students['high']
        medium_risk_students = risk_students['medium']
        low_risk_students = risk_students['low']
    
    return render_template('dashboard.html',
                         total_students=total_students,
                         high_risk_count=high_risk_count,
                         medium_risk_count=medium_risk_count,
                         low_risk_count=low_risk_count,
                         high_risk_students=high_risk_students,
                         medium_risk_students=medium_risk_students,
                         low_risk_students=low_risk_students)

@app.route('/predict')
def predict_page():
    return render_template('coming_soon.html', 
                         feature_name='Class Data Upload',
                         feature_description='Upload CSV or Excel files with multiple student records for batch analysis and risk assessment.')

@app.route('/validate')
def validate_models():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        import joblib
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from src.data.preprocessing import DataPreprocessor
        
        # Load and prepare data using the project's preprocessing pipeline
        data_path = 'data/student-por.csv'
        if not os.path.exists(data_path):
            return render_template('validate.html', 
                                 results={'Error': {'error': 'Dataset not found'}}, 
                                 total_students=0)
        
        preprocessor = DataPreprocessor()
        
        # Exclude G1 and G2 for realistic early prediction
        exclude_cols = ['G1', 'G2']
        X, y = preprocessor.preprocess_data(data_path, ',', exclude_cols=exclude_cols)
        
        # Create risk categories using the same method as training
        y_categories = preprocessor.create_risk_categories(y, method='quantile')
        
        # Encode labels using the same method
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_categories)
        
        # Scale features using the same method
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Get the actual risk category names from the label encoder
        risk_categories = label_encoder.classes_
        print(f"Risk categories: {risk_categories}")  # Debug info
        
        # Create mapping from encoded values to risk labels
        risk_mapping = {i: category for i, category in enumerate(risk_categories)}
        y_actual = [risk_mapping[label] for label in y_encoded]
        
        # Test models
        models = {
            'Decision Tree': 'saved_models/decision_tree_student_performance.pkl',
            'KNN': 'saved_models/knn_student_performance.pkl', 
            'SVM': 'saved_models/svm_student_performance.pkl'
        }
        
        results = {}
        for name, path in models.items():
            try:
                if not os.path.exists(path):
                    results[name] = {'error': f'Model file not found: {path}'}
                    continue
                
                # Load model using joblib (correct method)
                model_data = joblib.load(path)
                
                # Extract the actual model from the saved structure
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                else:
                    model = model_data
                
                # Make predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_scaled)
                else:
                    results[name] = {'error': 'Invalid model format'}
                    continue
                
                # Convert predictions to risk labels using the correct mapping
                pred_labels = []
                for pred in predictions:
                    if hasattr(pred, 'item'):
                        pred = pred.item()
                    pred_labels.append(risk_mapping.get(int(pred), 'Unknown'))
                
                # Calculate accuracy
                correct_predictions = sum(1 for actual, pred in zip(y_actual, pred_labels) if actual == pred)
                accuracy = correct_predictions / len(y_actual) if len(y_actual) > 0 else 0
                
                results[name] = {
                    'accuracy': round(accuracy * 100, 1),
                    'predictions': pred_labels,
                    'sample_comparisons': list(zip(y_actual[:10], pred_labels[:10]))
                }
                
            except Exception as e:
                results[name] = {'error': f'{type(e).__name__}: {str(e)}'}
        
        return render_template('validate.html', 
                             results=results, 
                             total_students=len(X))
        
    except ImportError as e:
        return render_template('validate.html', 
                             results={'Import Error': {'error': f'Missing dependency: {str(e)}'}}, 
                             total_students=0)
    except Exception as e:
        return render_template('validate.html', 
                             results={'System Error': {'error': f'Validation failed: {str(e)}'}}, 
                             total_students=0)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
