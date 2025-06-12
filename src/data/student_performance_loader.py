# src/data/student_performance_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class StudentPerformanceDataLoader:
    """Loader for the Portuguese student performance dataset"""
    
    def __init__(self):
        self.feature_descriptions = {
            # Student info
            'school': 'student\'s school (binary: "GP" or "MS")',
            'sex': 'student\'s sex (binary: "F" - female or "M" - male)',
            'age': 'student\'s age (numeric: from 15 to 22)',
            'address': 'student\'s home address type (binary: "U" - urban or "R" - rural)',
            'famsize': 'family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)',
            'Pstatus': 'parent\'s cohabitation status (binary: "T" - living together or "A" - apart)',
            
            # Parent info
            'Medu': 'mother\'s education (numeric: 0-4)',
            'Fedu': 'father\'s education (numeric: 0-4)',
            'Mjob': 'mother\'s job (nominal)',
            'Fjob': 'father\'s job (nominal)',
            
            # Student background
            'reason': 'reason to choose this school (nominal)',
            'guardian': 'student\'s guardian (nominal)',
            'traveltime': 'home to school travel time (numeric: 1-4)',
            'studytime': 'weekly study time (numeric: 1-4)',
            'failures': 'number of past class failures (numeric: n if 1<=n<3, else 4)',
            'schoolsup': 'extra educational support (binary: yes or no)',
            'famsup': 'family educational support (binary: yes or no)',
            'paid': 'extra paid classes (binary: yes or no)',
            'activities': 'extra-curricular activities (binary: yes or no)',
            'nursery': 'attended nursery school (binary: yes or no)',
            'higher': 'wants to take higher education (binary: yes or no)',
            'internet': 'Internet access at home (binary: yes or no)',
            'romantic': 'with a romantic relationship (binary: yes or no)',
            
            # Social life
            'famrel': 'quality of family relationships (numeric: from 1 - very bad to 5 - excellent)',
            'freetime': 'free time after school (numeric: from 1 - very low to 5 - very high)',
            'goout': 'going out with friends (numeric: from 1 - very low to 5 - very high)',
            'Dalc': 'workday alcohol consumption (numeric: from 1 - very low to 5 - very high)',
            'Walc': 'weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)',
            'health': 'current health status (numeric: from 1 - very bad to 5 - very good)',
            'absences': 'number of school absences (numeric: from 0 to 93)',
            
            # Grades
            'G1': 'first period grade (numeric: from 0 to 20)',
            'G2': 'second period grade (numeric: from 0 to 20)',
            'G3': 'final grade (numeric: from 0 to 20, output target)'
        }
        
    def load_data(self, filepath, delimiter=','): 
        """Load the student performance dataset"""
        try:
            data = pd.read_csv(filepath, delimiter=delimiter)
            print(f"Loaded dataset with shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_risk_categories(self, df, target_col='G3', include_g1_g2=False):
        """Create risk categories based on final grade"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Remove G1 and G2 if not including them (more realistic scenario)
        if not include_g1_g2 and 'G1' in df.columns and 'G2' in df.columns:
            print("Excluding G1 and G2 from features (more realistic early prediction)")
            df = df.drop(['G1', 'G2'], axis=1)
        
        # Create risk categories based on Portuguese grading system
        # 0-9: Fail (Critical Risk)
        # 10-11: Pass but at risk (High Risk)
        # 12-14: Satisfactory (Medium Risk)
        # 15-20: Good/Excellent (Low Risk)
        
        df['risk_level'] = pd.cut(
            df[target_col],
            bins=[-1, 9, 11, 14, 20],
            labels=['Critical', 'High', 'Medium', 'Low']
        )
        
        # Create binary pass/fail
        df['pass_fail'] = (df[target_col] >= 10).astype(int)
        
        # Create numeric risk score (inverse of grade)
        df['risk_score'] = 1 - (df[target_col] / 20)
        
        return df
    
    def prepare_features(self, df, target_col='risk_level'):
        """Prepare features for machine learning"""
        # Separate features and target
        if target_col in df.columns:
            X = df.drop([target_col, 'G3', 'risk_score', 'pass_fail'], 
                       axis=1, errors='ignore')
            y = df[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Convert categorical variables to numeric
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            # Use one-hot encoding for non-binary categoricals
            if X[col].nunique() > 2:
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(col, axis=1)
            else:
                # Binary encoding for binary variables
                X[col] = pd.Categorical(X[col]).codes
        
        return X, y
    
    def get_feature_info(self):
        """Get information about features"""
        return self.feature_descriptions
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Data split completed:")
        print(f"  Train set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test