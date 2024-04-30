import pandas as pd
import re
import logging
import sys
from pathlib import Path

# Setup paths and logging
FILE = Path(__file__).resolve()
SRC_ROOT = FILE.parent  # SRC directory
PROJECT_ROOT = SRC_ROOT.parent  # Project root directory
DATA_PATH = PROJECT_ROOT / 'data'  # Data directory

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))  # Add SRC to PATH if not already included

script_name = FILE.stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(SRC_ROOT / f'{script_name}.log'), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)


def load_and_clean_data(filepath):
    """
    Load data from a CSV file and filter necessary columns.

    Parameters:
    - filepath (Path): Path to the CSV file.

    Returns:
    - pd.DataFrame: Dataframe with columns 'class' and 'tweet' after initial cleanup.

    Raises:
    - FileNotFoundError: If the file path does not exist.
    - Exception: For other issues that might arise when loading the data.
    """
    try:
        df = pd.read_csv(filepath, names=['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])
        df = df[['class', 'tweet']]
        logging.info("Data loaded and cleaned successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise


def strip_all_entities(text):
    """
    Remove URLs, mentions, and any non-alphanumeric characters from the tweet text.

    Parameters:
    - text (str): The original tweet text.

    Returns:
    - str: Cleaned tweet text.
    """
    try:
        pattern = r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
        cleaned_text = ' '.join(re.sub(pattern, " ", text).split())
        logging.debug(f"Original text: {text}, Cleaned text: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        raise


def preprocess_and_save_data(input_filepath, output_filepath):
    """
    Preprocess the data by loading, cleaning, applying text transformations, and saving the cleaned data to a new CSV file.

    Parameters:
    - input_filepath (Path): Path to the dataset CSV file.
    - output_filepath (Path): Path where the cleaned data file will be saved.
    """
    try:
        df = load_and_clean_data(input_filepath)
        df['tweet'] = df['tweet'].apply(strip_all_entities)
        df.to_csv(output_filepath, index=False)
        logging.info(f"Data preprocessing completed and saved to {output_filepath}")
    except Exception as e:
        logging.error(f"Error during data preprocessing and saving: {e}")
        raise


# Example usage
if __name__ == "__main__":
    try:
        input_filepath = DATA_PATH / 'labeled_data.csv'
        output_filepath = DATA_PATH / 'preprocessed_labeled_data.csv'
        preprocess_and_save_data(input_filepath, output_filepath)
        logging.info("Preprocessing complete.")
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
