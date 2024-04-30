import os
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from utils.logger import get_logger
from configs import config

# Load configurations from a centralized configuration file
CFG = config.CFG
# Initialize a logger for monitoring and debugging
LOG = get_logger('hate_speech_predictor')


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation F1 score
    does not improve after a set number of epochs.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int): Number of epochs to wait before stopping after detecting no improvement.
            verbose (bool): Enables verbose output in the logging.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model file.
            trace_func (function): Function used to log the early stopping status.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_score_min = float('inf')

    def __call__(self, val_f1, model):
        """
        Call method that acts as a step during each epoch to check if early stopping conditions are met.

        Args:
            val_f1 (float): Current epoch's validation F1 score.
            model (torch.nn.Module): The model being trained.
        """
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        """
        Saves the model when the validation F1 score improves.

        Args:
            val_f1 (float): Validation F1 score that triggered the saving.
            model (torch.nn.Module): Model to save.
        """
        if self.verbose:
            self.trace_func(
                f'Validation F1 score increased ({self.val_score_min:.6f} --> {val_f1:.6f}). The model has been saved!')
        torch.save(model.state_dict(), self.path)
        self.val_score_min = val_f1


class HateSpeechPredictor:
    """
    Manages training and evaluation of a hate speech detection model.
    """

    def __init__(self, config, model, train_loader, val_loader):
        """
        Initializes the predictor object.

        Args:
            config (dict): Configuration object with settings and hyperparameters.
            model (torch.nn.Module): The model to train and evaluate.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        # Set up the loss function, optimizer, and early stopping mechanism.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=config['train']['learning_rate'])
        self.early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.001,
                                            path=os.path.join(config['paths']['model_path'], 'best_model.pt'),
                                            trace_func=LOG.info)

    def train(self):
        """
        Executes the training process with early stopping.
        """
        for epoch in range(self.config['train']['epochs']):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            LOG.info(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(self.train_loader)}')
            validation_f1 = self.evaluate()

            # Check for early stopping
            self.early_stopping(validation_f1, self.model)
            if self.early_stopping.early_stop:
                LOG.info('Early stopping triggered')
                break

    def evaluate(self):
        """
        Evaluates the model on the validation dataset.

        Returns:
            float: The F1 score from the validation dataset.
        """
        self.model.eval()
        total_loss, all_targets, all_predictions = 0, [], []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(inputs, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro',
                                                                         zero_division=0)
        LOG.info(f'Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
        return f1_score
