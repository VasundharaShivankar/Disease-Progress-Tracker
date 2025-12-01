"""
Health Plus - Advanced Medical AI Platform
Main application factory and configuration
"""

import os
import sys
from flask import Flask, request, g
from flask_cors import CORS
from flask_login import LoginManager
from flask_pymongo import PyMongo
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from config import get_config
from utils.logger import setup_logger
from database import init_db
from database import init_db

# Initialize extensions
login_manager = LoginManager()
mongo = PyMongo()

def create_app(config_name=None):
    """
    Application factory function

    Args:
        config_name (str): Configuration environment name

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    login_manager.init_app(app)
    mongo = init_db(app)

    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[app.config['RATELIMIT_DEFAULT']],
        storage_uri=app.config['RATELIMIT_STORAGE_URL'],
        strategy=app.config['RATELIMIT_STRATEGY']
    )

    # Initialize CSRF protection
    csrf = CSRFProtect(app)

    # Configure login manager
    login_manager.login_view = app.config['LOGIN_VIEW']
    login_manager.login_message = app.config['LOGIN_MESSAGE']
    login_manager.login_message_category = app.config['LOGIN_MESSAGE_CATEGORY']

    # Set up logging
    setup_logger()

    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.analysis import analysis_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(analysis_bp, url_prefix='/analysis')

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID for Flask-Login"""
        from models import User
        try:
            user_data = mongo.db.users.find_one({'_id': user_id})
            if user_data:
                return User(user_data)
        except Exception as e:
            app.logger.error(f"Error loading user {user_id}: {str(e)}")
        return None

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        from flask import render_template
        app.logger.warning(f"404 error: {error}")
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        from flask import render_template
        app.logger.error(f"500 error: {error}")
        return render_template('500.html'), 500

    @app.errorhandler(403)
    def forbidden_error(error):
        """Handle 403 errors"""
        from flask import render_template, flash, redirect, url_for
        flash('You do not have permission to access this resource.', 'error')
        return redirect(url_for('main.home'))

    @app.errorhandler(429)
    def ratelimit_error(error):
        """Handle rate limiting errors"""
        from flask import jsonify, render_template
        app.logger.warning(f"Rate limit exceeded: {error}")
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'error': 'Too many requests. Please try again later.'}), 429
        else:
            return render_template('429.html'), 429

    @app.errorhandler(400)
    def bad_request_error(error):
        """Handle 400 errors"""
        from flask import render_template, jsonify
        app.logger.warning(f"400 error: {error}")
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'error': 'Bad request'}), 400
        return render_template('400.html'), 400

    @app.errorhandler(422)
    def unprocessable_entity_error(error):
        """Handle 422 errors (validation errors)"""
        from flask import jsonify, render_template
        app.logger.warning(f"422 error: {error}")
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'error': 'Unprocessable entity'}), 422
        return render_template('422.html'), 422

    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring"""
        return {'status': 'healthy', 'timestamp': '2024-01-01T00:00:00Z'}

    # Context processor for template variables
    @app.context_processor
    def inject_template_vars():
        """Inject common variables into all templates"""
        return {
            'app_name': 'Health Plus',
            'current_year': 2024
        }

    app.logger.info(f"Health Plus application started in {config_name or 'development'} mode")

    return app

# Create app instance for gunicorn
app = create_app()

# Main entry point
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG']
    )

