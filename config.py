import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # MongoDB Configuration
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/health_plus_db'

    # Upload Configuration
    UPLOAD_FOLDER = 'uploads/'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Flask-Login Configuration
    LOGIN_VIEW = 'auth.login'
    LOGIN_MESSAGE = 'Please log in to access this page.'
    LOGIN_MESSAGE_CATEGORY = 'info'

    # CORS Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']

    # ML Model Configuration
    MODEL_PATH = 'skinmodel_vgg16.h5'
    IMG_SIZE = (224, 224)
    CLASS_LABELS = ['acne', 'dermatitis', 'hyperpigmentation', 'psoriasis', 'nail_psoriasis']

    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s'

    # Rate Limiting Configuration
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "memory://"
    RATELIMIT_STRATEGY = "fixed-window"

    # CSRF Protection Configuration
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = os.environ.get('WTF_CSRF_SECRET_KEY') or 'csrf-secret-key-change-in-production'
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour

    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    def __init__(self):
        super().__init__()
        # Override with environment variables for production
        self.SECRET_KEY = os.environ.get('SECRET_KEY')
        self.MONGO_URI = os.environ.get('MONGO_URI')

        # Validate required environment variables
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable is required in production")

        if not self.MONGO_URI:
            raise ValueError("MONGO_URI environment variable is required in production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MONGO_URI = 'mongodb://localhost:27017/test_health_plus_db'
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    return config.get(config_name, config['default'])
