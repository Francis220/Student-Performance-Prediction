from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    role = db.Column(db.String(50), nullable=False)  # 'educator', 'data_analyst', 'administrator'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class Student(db.Model):
    """Student information model"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    email = db.Column(db.String(120))
    enrollment_date = db.Column(db.Date)
    program = db.Column(db.String(100))
    
    # Relationships
    academic_records = db.relationship('AcademicRecord', backref='student', lazy=True)
    attendance_records = db.relationship('AttendanceRecord', backref='student', lazy=True)
    lms_activities = db.relationship('LMSActivity', backref='student', lazy=True)
    risk_predictions = db.relationship('RiskPrediction', backref='student', lazy=True)

class AcademicRecord(db.Model):
    """Academic performance records"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.String(50), nullable=False)
    course_name = db.Column(db.String(200))
    semester = db.Column(db.String(20))
    grade = db.Column(db.Float)
    credits = db.Column(db.Integer)
    assignment_completion_rate = db.Column(db.Float)
    quiz_average = db.Column(db.Float)
    midterm_score = db.Column(db.Float)
    
class AttendanceRecord(db.Model):
    """Student attendance records"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20))  # 'present', 'absent', 'late'
    duration_minutes = db.Column(db.Integer)

class LMSActivity(db.Model):
    """Learning Management System activity data"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.String(50), nullable=False)
    activity_date = db.Column(db.DateTime, nullable=False)
    activity_type = db.Column(db.String(50))  # 'login', 'assignment_view', 'resource_download', etc.
    duration_seconds = db.Column(db.Integer)
    page_views = db.Column(db.Integer)
    
class RiskPrediction(db.Model):
    """ML model predictions for student risk"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    model_name = db.Column(db.String(50), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20))  # 'Low', 'Medium', 'High', 'Critical'
    confidence_score = db.Column(db.Float)
    explanation = db.Column(db.Text)
    intervention_recommendations = db.Column(db.Text)