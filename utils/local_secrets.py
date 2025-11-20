"""
Complete Configuration for ProVe
"""

MONGODB_URL = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "prove_db"
LOG_FILENAME = "prove.log"
LOG_PATH = "./logs"

# Add missing endpoints
ENDPOINT = "http://localhost:5000"
API_KEY = ""

def get_database_config():
    return {
        'mongodb_url': MONGODB_URL,
        'db_name': MONGODB_DB_NAME
    }
