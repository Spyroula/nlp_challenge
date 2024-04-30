import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from torchsampler import ImbalancedDatasetSampler
from configs import config

# Load the configuration settings from config module.
CFG = config.CFG


class HateSpeechDataset(Dataset):
    """
    A PyTorch Dataset class for handling hate speech data. This class is responsible for preparing
    and tokenizing the data to be fed into a neural network.

    Attributes:
        dataframe (DataFrame): Source data frame containing the tweets and labels.
        tokenizer: Tokenizer instance used to convert text to tokens.
        max_len (int): Maximum length of the tokens.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the HateSpeechDataset instance.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the 'tweet' and 'class' columns.
            tokenizer: Tokenizer that converts text to tokens with attention masks.
            max_len (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.texts = dataframe['tweet']
        self.labels = dataframe['class']
        self.max_len = max_len

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.texts)

    def __getitem__(self, index):
        """
        Retrieves an item by its index and returns its tokenized form.

        Args:
            index (int): Index of the item in the dataset.

        Returns:
            dict: A dictionary containing the input IDs, attention masks, and labels.
        """
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_labels(self):
        """
        Returns all labels from the dataset. Useful for some sampling techniques which require access to all labels.

        Returns:
            pd.Series: Series containing all the labels.
        """
        return self.labels


class HateSpeechDataLoader:
    """
    DataLoader class to handle the loading and preprocessing of hate speech data.

    Attributes:
        file_path (str): Path to the CSV file containing the data.
        tokenizer: Tokenizer instance for tokenizing data.
        max_len (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for training data.
        valid_batch_size (int): Batch size for validation data.
    """

    def __init__(self, file_path, tokenizer, max_len, batch_size, valid_batch_size):
        """
        Initializes the HateSpeechDataLoader instance.

        Args:
            file_path (str): Path to the dataset file.
            tokenizer: Tokenizer for processing text data.
            max_len (int): Maximum length for tokenization.
            batch_size (int): Batch size for the training set.
            valid_batch_size (int): Batch size for the validation set.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

    @staticmethod
    def clean_text(text):
        """
        Static method to clean text data by removing URLs, HTML tags, and special characters.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^A-Za-z\s]+', '', text)  # Remove special characters and numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def load_data(self):
        """
        Loads data from a CSV file, cleans it, splits it into training and testing datasets,
        and returns corresponding data loaders.

        Returns:
            tuple: A tuple containing training and testing DataLoader instances.
        """
        df = pd.read_csv(self.file_path)
        df['tweet'] = df['tweet'].apply(self.clean_text)
        train_data = df.sample(frac=0.8, random_state=200)  # Randomly sample 80% of the data for training.
        test_data = df.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)

        # Create dataset instances for training and testing.
        training_set = HateSpeechDataset(train_data, self.tokenizer, self.max_len)
        testing_set = HateSpeechDataset(test_data, self.tokenizer, self.max_len)

        # Parameters for PyTorch DataLoader
        train_params = {'batch_size': self.batch_size, 'sampler': ImbalancedDatasetSampler(training_set),
                        'num_workers': 0}
        test_params = {'batch_size': self.valid_batch_size, 'shuffle': True, 'num_workers': 0}

        # Creating the DataLoader instances for training and testing datasets.
        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        return training_loader, testing_loader
