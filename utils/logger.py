"""
Logger module for ProVe
"""
import logging
import os
from .local_secrets import LOG_FILENAME, LOG_PATH

# Create logs directory if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Setup logger
logger = logging.getLogger('prove')
logger.setLevel(logging.INFO)

# Create file handler
log_file = os.path.join(LOG_PATH, LOG_FILENAME)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    return logger
