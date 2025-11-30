import logging
import logging.handlers
from pathlib import Path
import sys
from config import get_config

def setup_logger(name='health_plus', config=None):
    """
    Set up comprehensive logging for the application

    Args:
        name (str): Logger name
        config: Configuration object

    Returns:
        logging.Logger: Configured logger instance
    """
    if config is None:
        config = get_config()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(config.LOG_FORMAT)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )

    # File handler with rotation
    log_file = Path(config.LOG_FILE)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_request_logger():
    """Get logger specifically for request logging"""
    return logging.getLogger('health_plus.requests')

def get_error_logger():
    """Get logger specifically for error logging"""
    return logging.getLogger('health_plus.errors')

def log_request(logger, method, path, user_id=None, duration=None):
    """Log HTTP request details"""
    user_info = f" (User: {user_id})" if user_id else ""
    duration_info = f" in {duration:.2f}s" if duration else ""
    logger.info(f"Request: {method} {path}{user_info}{duration_info}")

def log_error(logger, error, context=None):
    """Log error with context"""
    context_str = f" Context: {context}" if context else ""
    logger.error(f"Error: {str(error)}{context_str}", exc_info=True)

def log_model_prediction(logger, model_name, prediction, confidence, user_id=None):
    """Log ML model predictions"""
    user_info = f" (User: {user_id})" if user_id else ""
    logger.info(f"Model: {model_name} predicted {prediction} with {confidence} confidence{user_info}")

def log_file_upload(logger, filename, file_size, user_id=None):
    """Log file upload events"""
    user_info = f" (User: {user_id})" if user_id else ""
    logger.info(f"File upload: {filename} ({file_size} bytes){user_info}")

class RequestLogger:
    """Context manager for request logging"""

    def __init__(self, logger, method, path, user_id=None):
        self.logger = logger
        self.method = method
        self.path = path
        self.user_id = user_id
        self.start_time = None

    def __enter__(self):
        self.start_time = logging.time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = logging.time.time() - self.start_time
        log_request(self.logger, self.method, self.path, self.user_id, duration)
        if exc_type:
            log_error(self.logger, exc_val, f"Exception in request {self.method} {self.path}")
