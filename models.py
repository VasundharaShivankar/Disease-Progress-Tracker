"""
Data models for Health Plus application
"""

from flask_login import UserMixin
from datetime import datetime
import uuid

class User(UserMixin):
    """User model for Flask-Login"""

    def __init__(self, id, email, password_hash, name=None, created_at=None):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.name = name
        self.created_at = created_at

    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            '_id': self.id,
            'email': self.email,
            'password_hash': self.password_hash,
            'name': self.name,
            'created_at': self.created_at
        }

class FileUpload:
    """Model for file upload tracking"""

    def __init__(self, user_id, filename, original_filename, file_path, file_size, file_type):
        self.user_id = user_id
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.file_type = file_type
        self.uploaded_at = datetime.utcnow()

    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'uploaded_at': self.uploaded_at
        }

class AnalysisResult:
    """Model for analysis results tracking"""

    def __init__(self, user_id, analysis_type, result_data):
        self.user_id = user_id
        self.analysis_type = analysis_type  # 'skin', 'spine', 'progress'
        self.result_data = result_data
        self.created_at = datetime.utcnow()

    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            'user_id': self.user_id,
            'analysis_type': self.analysis_type,
            'result_data': self.result_data,
            'created_at': self.created_at
        }

class UserActivity:
    """Model for tracking user activity"""

    def __init__(self, user_id, activity_type, details=None):
        self.user_id = user_id
        self.activity_type = activity_type  # 'login', 'analysis', 'upload', etc.
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            'user_id': self.user_id,
            'activity_type': self.activity_type,
            'details': self.details,
            'timestamp': self.timestamp
        }

# Initialize MongoDB connection
from flask_pymongo import PyMongo
mongo = PyMongo()
