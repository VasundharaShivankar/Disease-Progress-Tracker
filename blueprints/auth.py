from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, login_required, logout_user, current_user
from flask_wtf.csrf import generate_csrf
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import mongo
from models import User
from utils.logger import get_request_logger, log_request, log_error
from utils.validators import validate_email, validate_password, validate_name
import re

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
logger = get_request_logger()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip()
            password = request.form.get('password')
            action = request.form.get('action')

            # Validate input
            if not email or not password:
                flash('Email and password are required.', 'error')
                return render_template('login.html')

            if not validate_email(email):
                flash('Please enter a valid email address.', 'error')
                return render_template('login.html')

            if action == 'login':
                # Login logic
                user_data = mongo.db.users.find_one({'email': email.lower()})
                if user_data and check_password_hash(user_data['password_hash'], password):
                    user = User(
                        id=str(user_data['_id']),
                        email=user_data['email'],
                        password_hash=user_data['password_hash'],
                        name=user_data.get('name'),
                        created_at=user_data.get('created_at')
                    )
                    login_user(user)
                    log_request(logger, 'POST', '/login', user.id)
                    flash('Logged in successfully!', 'success')

                    next_page = request.args.get('next')
                    if next_page:
                        return redirect(next_page)
                    return redirect(url_for('main.home'))
                else:
                    log_error(logger, "Invalid login attempt", f"Email: {email}")
                    flash('Invalid email or password.', 'error')

            elif action == 'signup':
                # Signup logic
                name = request.form.get('name', '').strip()

                # Validate signup data
                if not validate_name(name):
                    flash('Name must be 2-50 characters long and contain only letters, spaces, and hyphens.', 'error')
                    return render_template('login.html')

                if not validate_password(password):
                    flash('Password must be at least 8 characters long and contain uppercase, lowercase, and numbers.', 'error')
                    return render_template('login.html')

                # Check if user exists
                existing_user = mongo.db.users.find_one({'email': email.lower()})
                if existing_user:
                    flash('Email already registered.', 'error')
                    return render_template('login.html')

                # Create new user
                password_hash = generate_password_hash(password)
                user_doc = {
                    'email': email.lower(),
                    'password_hash': password_hash,
                    'name': name,
                    'created_at': datetime.utcnow(),
                    'last_login': datetime.utcnow()
                }

                result = mongo.db.users.insert_one(user_doc)
                user = User(
                    id=str(result.inserted_id),
                    email=email,
                    password_hash=password_hash,
                    name=name,
                    created_at=user_doc['created_at']
                )

                login_user(user)
                log_request(logger, 'POST', '/login', user.id)
                flash('Account created successfully!', 'success')
                return redirect(url_for('main.home'))

        except Exception as e:
            log_error(logger, e, "Login/Signup error")
            flash('An error occurred. Please try again.', 'error')

    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    try:
        log_request(logger, 'GET', '/logout', current_user.id)
        logout_user()
        flash('Logged out successfully!', 'success')
    except Exception as e:
        log_error(logger, e, "Logout error")
        flash('An error occurred during logout.', 'error')

    return redirect(url_for('main.home'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    try:
        user_data = mongo.db.users.find_one({'_id': current_user.id})
        if not user_data:
            flash('User not found.', 'error')
            return redirect(url_for('main.home'))

        return render_template('profile.html', user=user_data)
    except Exception as e:
        log_error(logger, e, f"Profile access error for user {current_user.id}")
        flash('An error occurred loading your profile.', 'error')
        return redirect(url_for('main.home'))
