import logging
import os

def get_logger(name):
    """Create and return a logger object."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set to DEBUG to capture all types of logs

    # Formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler to save logs to a file
    file_handler = logging.FileHandler(f'../logs/{name}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler to print to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
