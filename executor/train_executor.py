import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import config
from trainer.model import RobertaClassifier
from datahandler import HateSpeechDataLoader
from trainer.train import HateSpeechPredictor
from transformers import RobertaTokenizer


def train_model():
    """
    This function sets up the training environment, loads the data, initializes the model,
    and begins the training process.
    """
    # Load the configuration from the 'config' module
    CFG = config.CFG

    # Set the CUDA device to be visible, if using GPU for training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU

    # Initialize the tokenizer with the specified model type from configuration
    tokenizer = RobertaTokenizer.from_pretrained(CFG['model']['tokenizer'])

    # Create an instance of the data loader with specified configurations
    data_loader = HateSpeechDataLoader(
        os.path.join(CFG['paths']['data_path'], CFG['data']['data_file']),
        tokenizer,
        CFG['data']['max_len'],
        CFG['train']['batch_size'],
        CFG['train']['valid_batch_size']
    )

    # Load training and validation data
    train_loader, val_loader = data_loader.load_data()

    # Initialize the model with the number of classes defined in the configuration
    model = RobertaClassifier(num_classes=CFG['data']['num_classes'])

    # Initialize the predictor, passing the configuration and data loaders
    predictor = HateSpeechPredictor(CFG, model, train_loader, val_loader)

    # Start the training process
    predictor.train()


if __name__ == "__main__":
    train_model()
