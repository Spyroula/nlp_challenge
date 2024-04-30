import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = {
    "paths": {
        "data_path": os.path.join(BASE_DIR, 'data'),
        "model_path": os.path.join(BASE_DIR, 'models'),
        "log_path": os.path.join(BASE_DIR, 'logs'),
        "logging_config": os.path.join(BASE_DIR, 'configs', 'logging_config.yaml')
    },
    "data": {
        "data_file": "labeled_data.csv",
        "max_len": 256,
        "num_classes": 3
    },
    "train": {
        "batch_size": 32,
        "valid_batch_size": 16,
        "learning_rate": 1e-05,
        "epochs": 10,
        "optimizer": "adam",
        "metrics": ["precision", "recall", "f1_score"]
    },
    "model": {
        "tokenizer": "roberta-base",
        "hidden_size": 768,
        "pre_classifier": {
            "in_features": 768,
            "out_features": 768,
            "dropout": 0.3
        },
        "classifier": {
            "in_features": 768,
            "out_features": 3
        }
    },
    "labels": {
        0: "hate speech",
        1: "offensive language",
        2: "neither"
    }
}
