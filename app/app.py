import os
from flask import Flask, request, jsonify
import torch
from transformers import RobertaTokenizer
from trainer.model import RobertaClassifier
from configs.config import CFG  # Ensure this import path is correct

app = Flask(__name__)

MODEL_PATH = os.path.join(CFG['paths']['model_path'], 'best_model.pt')

# Load the tokenizer and model according to specified configurations
tokenizer = RobertaTokenizer.from_pretrained(CFG['model']['tokenizer'], truncation=True, do_lower_case=True)
model = RobertaClassifier(num_classes=CFG['data']['num_classes'])
# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set the model to evaluation mode


@app.route('/predict', methods=['POST'])
def predict():
    """
    Define the prediction endpoint which takes text input and returns model predictions with confidence.
    """
    data = request.get_json(force=True)
    text = data['text']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CFG['data']['max_len'])

    with torch.no_grad():
        logits = model(**inputs)

    predicted_idx = logits.argmax(-1).item()
    response = {
        'prediction': CFG['labels'][predicted_idx],
        'confidence': torch.softmax(logits, dim=-1).max().item()
        # Providing the highest softmax probability as confidence
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
