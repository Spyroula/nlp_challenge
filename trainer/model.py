import torch.nn as nn
from transformers import RobertaModel


class RobertaClassifier(nn.Module):
    """
    This class defines a classifier that uses the RoBERTa model as its backbone. It is designed for text
    classification tasks, leveraging the pre-trained 'roberta-base' model, followed by a dropout layer and
    a linear layer for the final classification.

    Attributes:
        num_classes (int): Number of classes for the classification task.
    """

    def __init__(self, num_classes):
        """
        Initializes the RobertaClassifier model with a specified number of output classes.

        Args: num_classes (int): The number of classes in the classification task. This affects
        the output size of the final linear layer.
        """
        super(RobertaClassifier, self).__init__()
        # Load a pre-trained RoBERTa model from the Hugging Face transformers library.
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Dropout layer to reduce over fitting by randomly setting input units to 0 during training.
        self.dropout = nn.Dropout(0.3)

        # A fully connected layer that maps the output of the RoBERTa model to the number of classes.
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            torch.Tensor: Logits from the classifier, representing the model predictions.
        """
        # Pass input through RoBERTa model; extract the last hidden state of the first token (typically [CLS]) from
        # the sequence.
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]  # Extract the embeddings of the [CLS] token for classification.

        # Apply dropout to the [CLS] token embeddings to prevent over fitting.
        sequence_output = self.dropout(sequence_output)

        # Pass the dropout-applied embeddings through the classifier to obtain logits for each class.
        logits = self.classifier(sequence_output)

        return logits
