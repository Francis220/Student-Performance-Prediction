import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, filepath, file_type='csv'):
        """Load data from various file formats"""
        if file_type == 'csv':
            return pd.read_csv(filepath)
        elif file_type == 'excel':
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def create_features(self, df):
        """Engineer features from raw data"""
        features = pd.DataFrame()
        
        # Academic features
        if 'grade' in df.columns:
            features['avg_grade'] = df.groupby('student_id')['grade'].mean()
            features['grade_trend'] = self._calculate_trend(df, 'grade')
            features['grade_volatility'] = df.groupby('student_id')['grade'].std()
        
        # Attendance features
        if 'attendance_rate' in df.columns:
            features['attendance_rate'] = df.groupby('student_id')['attendance_rate'].mean()
            features['attendance_consistency'] = df.groupby('student_id')['attendance_rate'].std()
            features['recent_attendance'] = self._get_recent_metric(df, 'attendance_rate', days=30)
        
        # LMS activity features
        if 'lms_logins' in df.columns:
            features['avg_lms_logins'] = df.groupby('student_id')['lms_logins'].mean()
            features['lms_engagement_score'] = self._calculate_lms_engagement(df)
            features['weekend_activity_ratio'] = self._calculate_weekend_activity(df)
        
        # Assignment features
        if 'assignment_completion_rate' in df.columns:
            features['assignment_completion_avg'] = df.groupby('student_id')['assignment_completion_rate'].mean()
            features['late_submission_rate'] = self._calculate_late_submission_rate(df)
        
        return features
    
    def _calculate_trend(self, df, column, window=3):
        """Calculate trend using moving average"""
        grouped = df.groupby('student_id')[column]
        return grouped.rolling(window=window).mean().groupby('student_id').last()
    
    def _get_recent_metric(self, df, column, days=30):
        """Get recent metric values within specified days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        recent_df = df[df['date'] >= cutoff_date]
        return recent_df.groupby('student_id')[column].mean()
    
    def _calculate_lms_engagement(self, df):
        """Calculate composite LMS engagement score"""
        engagement_score = (
            df['lms_logins'] * 0.3 +
            df['page_views'] * 0.2 +
            df['resource_downloads'] * 0.3 +
            df['discussion_posts'] * 0.2
        )
        return engagement_score.groupby(df['student_id']).mean()
    
    def _calculate_weekend_activity(self, df):
        """Calculate ratio of weekend to weekday activity"""
        df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek.isin([5, 6])
        weekend_activity = df[df['is_weekend']].groupby('student_id')['lms_logins'].sum()
        weekday_activity = df[~df['is_weekend']].groupby('student_id')['lms_logins'].sum()
        return weekend_activity / (weekday_activity + 1)  # Add 1 to avoid division by zero
    
    def _calculate_late_submission_rate(self, df):
        """Calculate rate of late submissions"""
        if 'submission_status' in df.columns:
            late_submissions = df[df['submission_status'] == 'late'].groupby('student_id').size()
            total_submissions = df.groupby('student_id').size()
            return late_submissions / total_submissions
        return pd.Series()
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with mean
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Impute categorical columns with mode
        for col in categorical_columns:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        else:
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        return df
    
    def create_risk_labels(self, df, target_column='final_grade'):
        """Create risk labels based on performance"""
        if target_column in df.columns:
            # Define risk thresholds
            df['risk_level'] = pd.cut(
                df[target_column],
                bins=[0, 50, 65, 80, 100],
                labels=['Critical', 'High', 'Medium', 'Low']
            )
        return df