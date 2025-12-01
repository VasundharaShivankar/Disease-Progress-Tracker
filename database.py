from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    """Initialize the database with the Flask app"""
    mongo.init_app(app)
    return mongo
