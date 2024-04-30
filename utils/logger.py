import os
import logging


def get_logger(name):
    """
    Creates and configures a logger.

    Args:
        name (str): Name of the logger, usually the module name where it's called from.

    Returns:
        logging.Logger: Configured logger with file and console handlers.
    """
    # Define the directory where log files will be stored
    log_directory = '../logs'  # Path can be adjusted based on the project structure

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)  # Ensures creation of the log directory

    # Construct the full path where the log file will be saved
    log_path = os.path.join(log_directory, f'{name}.log')

    # Create a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Define the format for the logging messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)  # Set the formatter for the file handler
    logger.addHandler(file_handler)  # Add the file handler to the logger

    # Create a stream handler to output logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)  # Set the formatter for the stream handler
    logger.addHandler(stream_handler)  # Add the stream handler to the logger

    return logger
