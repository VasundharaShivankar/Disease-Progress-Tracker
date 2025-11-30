import re
from werkzeug.utils import secure_filename
import os

def validate_email(email):
    """Validate email format"""
    if not email:
        return False

    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def validate_password(password):
    """Validate password strength"""
    if not password or len(password) < 8:
        return False

    # Check for at least one uppercase, one lowercase, and one digit
    has_upper = re.search(r'[A-Z]', password)
    has_lower = re.search(r'[a-z]', password)
    has_digit = re.search(r'\d', password)

    return all([has_upper, has_lower, has_digit])

def validate_name(name):
    """Validate name format"""
    if not name or len(name) < 2 or len(name) > 50:
        return False

    # Allow letters, spaces, hyphens, and apostrophes
    name_regex = r"^[a-zA-Z\s\-']+$"
    return re.match(name_regex, name.strip()) is not None

def validate_file_upload(file, allowed_extensions=None, max_size=None):
    """Validate file upload"""
    if not file or not file.filename:
        return False, "No file selected"

    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

    if max_size is None:
        max_size = 10 * 1024 * 1024  # 10MB

    # Check filename security
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"

    # Check file extension
    file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if file_ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > max_size:
        return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"

    return True, filename

def sanitize_input(text):
    """Sanitize text input to prevent XSS"""
    if not text:
        return ""

    # Remove potentially dangerous characters
    text = re.sub(r'[<>]', '', text)

    # Trim whitespace
    return text.strip()

def validate_mongo_object_id(oid):
    """Validate MongoDB ObjectId format"""
    if not oid:
        return False

    object_id_regex = r'^[a-fA-F0-9]{24}$'
    return re.match(object_id_regex, str(oid)) is not None
