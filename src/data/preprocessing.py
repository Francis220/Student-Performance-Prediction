import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def _validate_and_clean_data(self, df):
        """Validate and clean data to prevent numerical issues"""
        df_clean = df.copy()
        
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values appropriately
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                # Use median for numeric columns
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    # If median is NaN, use 0
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna(median_val)
        
        # Clip extreme values to prevent overflow
        for col in numeric_cols:
            if col in df_clean.columns:
                # Clip to reasonable range based on data type
                col_min = df_clean[col].quantile(0.01)
                col_max = df_clean[col].quantile(0.99)
                if col_max > col_min:  # Ensure valid range
                    df_clean[col] = np.clip(df_clean[col], col_min, col_max)
        
        return df_clean
        
    def create_advanced_features(self, df):
        
        df_enhanced = df.copy()
        
        # Validate and clean input data first
        df_enhanced = self._validate_and_clean_data(df_enhanced)
        
        # Parent education features (safe operations)
        df_enhanced['parent_education_avg'] = (df_enhanced['Medu'] + df_enhanced['Fedu']) / 2
        df_enhanced['parent_education_diff'] = np.abs(df_enhanced['Medu'] - df_enhanced['Fedu'])
        df_enhanced['parent_education_max'] = df_enhanced[['Medu', 'Fedu']].max(axis=1)
        
        # Alcohol features with safe division
        df_enhanced['alcohol_total'] = df_enhanced['Dalc'] + df_enhanced['Walc']
        # Use safe division with epsilon to prevent division by zero
        epsilon = 1e-8
        dalc_safe = np.maximum(df_enhanced['Dalc'], epsilon)
        ratio = df_enhanced['Walc'] / dalc_safe
        df_enhanced['alcohol_ratio'] = np.clip(ratio, 0, 10)
        df_enhanced['high_alcohol'] = (df_enhanced['alcohol_total'] > 6).astype(int)
        
        # Social features with safe operations
        df_enhanced['social_time'] = df_enhanced['goout'] + df_enhanced['freetime']
        goout_safe = np.maximum(df_enhanced['goout'], epsilon)
        study_social_ratio = df_enhanced['studytime'] / goout_safe
        df_enhanced['study_social_ratio'] = np.clip(study_social_ratio, 0, 10)
        
        # Support features
        schoolsup_val = df_enhanced['schoolsup'].map({'yes': 1, 'no': 0}).fillna(0)
        famsup_val = df_enhanced['famsup'].map({'yes': 1, 'no': 0}).fillna(0)
        df_enhanced['academic_support'] = schoolsup_val + famsup_val
        
        # Family stability (safe calculation)
        pstatus_val = df_enhanced['Pstatus'].map({'T': 1, 'A': 0}).fillna(0)
        family_stability = df_enhanced['famrel'] * (2 - pstatus_val)
        df_enhanced['family_stability'] = np.clip(family_stability, 0, 10)
        
        # Motivation score
        higher_val = df_enhanced['higher'].map({'yes': 1, 'no': 0}).fillna(0)
        activities_val = df_enhanced['activities'].map({'yes': 1, 'no': 0}).fillna(0)
        df_enhanced['motivation_score'] = higher_val * 2 + activities_val
        
        # Risk composite with clipping
        failure_penalty = df_enhanced['failures'] * 2
        support_bonus = df_enhanced['academic_support']
        motivation_bonus = df_enhanced['motivation_score']
        health_penalty = (5 - df_enhanced['health']) * 0.5
        absence_penalty = df_enhanced['absences'] * 0.1
        
        risk_composite = (failure_penalty + health_penalty + absence_penalty - 
                         support_bonus - motivation_bonus)
        df_enhanced['risk_composite'] = np.clip(risk_composite, -10, 20)
        
        # Study efficiency with safe division
        absences_safe = np.maximum(df_enhanced['absences'], 1)
        study_efficiency = (df_enhanced['studytime'] * df_enhanced['health']) / absences_safe
        df_enhanced['study_efficiency'] = np.clip(study_efficiency, 0, 50)
        
        # Home advantage
        internet_val = df_enhanced['internet'].map({'yes': 1, 'no': 0}).fillna(0)
        famsup_val = df_enhanced['famsup'].map({'yes': 1, 'no': 0}).fillna(0)
        famrel_bonus = (df_enhanced['famrel'] > 3).astype(int)
        df_enhanced['home_advantage'] = internet_val + famsup_val + famrel_bonus
        
        # Categorical age groups - handle edge cases
        try:
            age_groups = pd.cut(df_enhanced['age'], bins=[14, 16, 18, 22], 
                              labels=['young', 'middle', 'old'])
            df_enhanced['age_group'] = age_groups.astype(str)  # Convert to string to avoid categorical issues
        except Exception:
            # Fallback if cut fails
            df_enhanced['age_group'] = 'middle'
        
        # Categorical absence levels - handle edge cases  
        try:
            absence_groups = pd.cut(df_enhanced['absences'], bins=[-1, 0, 5, 15, 100], 
                                  labels=['none', 'low', 'medium', 'high'])
            df_enhanced['absence_level'] = absence_groups.astype(str)  # Convert to string to avoid categorical issues
        except Exception:
            # Fallback if cut fails
            df_enhanced['absence_level'] = 'low'
        
        # Final validation
        df_enhanced = self._validate_and_clean_data(df_enhanced)
        
        return df_enhanced
    
    def advanced_encoding(self, df, is_training=True):
        
        df_processed = df.copy()
        
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if is_training:
                    # Handle NaN values before encoding and include 'unknown' in training
                    df_processed[col] = df_processed[col].fillna('unknown')
                    
                    encoder = LabelEncoder()
                    # Ensure 'unknown' is in the classes
                    unique_vals = list(df_processed[col].unique())
                    if 'unknown' not in unique_vals:
                        unique_vals.append('unknown')
                    
                    # Fit encoder on all possible values including 'unknown'
                    encoder.fit(unique_vals)
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    if col in self.encoders:
                        # Handle NaN values
                        df_processed[col] = df_processed[col].fillna('unknown')
                        
                        # Handle unseen categories
                        unique_vals = df_processed[col].unique()
                        known_vals = self.encoders[col].classes_
                        unknown_mask = ~np.isin(df_processed[col], known_vals)
                        
                        # Replace unknown values with 'unknown' if it exists, otherwise most frequent
                        if unknown_mask.any():
                            if 'unknown' in known_vals:
                                df_processed.loc[unknown_mask, col] = 'unknown'
                            else:
                                most_frequent = known_vals[0]  # Use first class as default
                                df_processed.loc[unknown_mask, col] = most_frequent
                        
                        df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
                    else:
                        df_processed[col] = 0
        
        return df_processed
    
    def smart_scaling(self, df, is_training=True):
        
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        # Scale all numeric columns to prevent issues
        if is_training:
            scaler = StandardScaler()
            # Fit only on columns that have variance > 0
            cols_to_scale = []
            for col in numeric_cols:
                if df_scaled[col].std() > 1e-8:  # Only scale if there's variance
                    cols_to_scale.append(col)
            
            if cols_to_scale:
                df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
                self.scalers['standard'] = scaler
                self.scalers['cols_to_scale'] = cols_to_scale
        else:
            if 'standard' in self.scalers and 'cols_to_scale' in self.scalers:
                cols_to_scale = self.scalers['cols_to_scale']
                # Only scale columns that exist in current df
                existing_cols = [col for col in cols_to_scale if col in df_scaled.columns]
                if existing_cols:
                    df_scaled[existing_cols] = self.scalers['standard'].transform(df_scaled[existing_cols])
        
        # Final check for any remaining problematic values
        df_scaled = df_scaled.replace([np.inf, -np.inf], 0)
        df_scaled = df_scaled.fillna(0)
        
        return df_scaled
    
    def remove_outliers(self, df, threshold=3):
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        outlier_mask = np.ones(len(df_clean), dtype=bool)
        
        for col in numeric_cols:
            if df_clean[col].std() > 1e-8:  # Only check columns with variance
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                # Use finite values only
                valid_z_scores = np.isfinite(z_scores)
                col_mask = (z_scores < threshold) | ~valid_z_scores
                outlier_mask = outlier_mask & col_mask
        
        return df_clean[outlier_mask]
    
    def preprocess_data(self, filepath: str, delimiter: str = ',', target_col: str = 'G3',
                       exclude_cols: list = None, remove_outliers: bool = True):
        
        df = pd.read_csv(filepath, delimiter=delimiter)
        
        # Initial data validation
        print(f"Initial data shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        if exclude_cols:
            cols_to_drop = [col for col in exclude_cols if col in df.columns]
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns: {cols_to_drop}")
        
        # Validate target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        df_enhanced = self.create_advanced_features(df)
        
        if remove_outliers:
            original_size = len(df_enhanced)
            df_enhanced = self.remove_outliers(df_enhanced)
            print(f"Removed {original_size - len(df_enhanced)} outliers")
        
        # Separate target and features
        y = df_enhanced[target_col]
        X = df_enhanced.drop(columns=[target_col])
        
        # Encode categorical variables
        X_encoded = self.advanced_encoding(X, is_training=True)
        
        # Scale features
        X_scaled = self.smart_scaling(X_encoded, is_training=True)
        
        # Select only numeric features for final dataset
        X_final = X_scaled.select_dtypes(include=[np.number])
        
        # Final validation
        print(f"Final feature shape: {X_final.shape}")
        print(f"Features: {X_final.columns.tolist()}")
        
        # Check for any remaining issues
        if X_final.isnull().any().any():
            print("Warning: NaN values found in final features, filling with 0")
            X_final = X_final.fillna(0)
        
        if np.isinf(X_final.values).any():
            print("Warning: Infinite values found in final features, replacing with finite values")
            X_final = X_final.replace([np.inf, -np.inf], 0)
        
        self.feature_names = X_final.columns.tolist()
        
        return X_final, y
    
    def create_risk_categories(self, grades, method='quantile'):
        
        # Validate grades
        grades_clean = grades.dropna()
        if len(grades_clean) == 0:
            raise ValueError("No valid grades found")
        
        try:
            if method == 'quantile':
                categories = pd.qcut(grades_clean, q=4, labels=['Critical', 'High', 'Medium', 'Low'], 
                                   duplicates='drop')
            elif method == 'threshold':
                categories = pd.cut(grades_clean, bins=[0, 8, 12, 15, 20], 
                                  labels=['Critical', 'High', 'Medium', 'Low'])
            else:
                categories = pd.qcut(grades_clean, q=4, labels=['Critical', 'High', 'Medium', 'Low'],
                                   duplicates='drop')
            
            # Reindex to match original grades
            result = pd.Series(index=grades.index, dtype='category')
            result.loc[grades_clean.index] = categories
            result = result.fillna('Medium')  # Fill NaN with default category
            
        except Exception as e:
            print(f"Warning: Error in risk categorization ({e}), using simple binning")
            # Fallback to simple binning
            result = pd.cut(grades.fillna(grades.median()), 
                          bins=4, labels=['Critical', 'High', 'Medium', 'Low'])
        
        return result 