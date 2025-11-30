from flask import Blueprint, render_template, send_from_directory
from flask_login import login_required, current_user
from utils.logger import get_request_logger, log_request

main_bp = Blueprint('main', __name__)
logger = get_request_logger()

@main_bp.route('/')
def home():
    """Home page"""
    log_request(logger, 'GET', '/')
    return render_template('index.html')

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    log_request(logger, 'GET', f'/uploads/{filename}')
    return send_from_directory('uploads', filename)

@main_bp.route('/about')
def about():
    """About page"""
    log_request(logger, 'GET', '/about')
    return render_template('about.html')

@main_bp.route('/contact')
def contact():
    """Contact page"""
    log_request(logger, 'GET', '/contact')
    return render_template('contact.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    log_request(logger, 'GET', '/dashboard', current_user.id)
    return render_template('dashboard.html')

@main_bp.route('/privacy')
def privacy():
    """Privacy policy page"""
    log_request(logger, 'GET', '/privacy')
    return render_template('privacy.html')

@main_bp.route('/terms')
def terms():
    """Terms of service page"""
    log_request(logger, 'GET', '/terms')
    return render_template('terms.html')
