# NLP Project for Hate Speech Detection

This project utilizes Natural Language Processing (NLP) to identify and classify hate speech in textual data. The project is structured to support training and deployment phases with robust logging and configuration management.

## Project Structure
```
nlp_project
├── app                     # Flask application for deploying the model
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Python dependencies required for the app
│   └── Dockerfile          # Docker container configuration
├── configs                 # Configuration files
│   ├── config.py           # Main configuration file
│   └── __init__.py         # Initialization file for configs module
├── data                    # Data directory
│   └── labeled_data.csv    # Labeled dataset
├── datahandler             # Module for data loading and preprocessing
│   ├── __init__.py
│   └── datahandler.py      # Handles data loading and cleaning
├── executor                # Execution logic for training and prediction
│   ├── __init__.py
│   └── train_executor.py   # Orchestrates the training process
├── logs                    # Log files directory
├── models                  # Trained model storage
│   └── best_model.pt       # Best model saved after training
├── trainer                 # Training logic and model definition
│   ├── __init__.py
│   ├── model.py            # Model architecture
│   └── train.py            # Training procedures
├── utils                   # Utility functions
│   ├── __init__.py
│   └── logger.py           # Custom logger for logging across the project
├── README.md               # Project overview and documentation
├── Pipfile                 # Pipenv file listing package dependencies
└── Pipfile.lock            # Lock file to ensure deterministic builds
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Spyroula/nlp_challenge.git
   cd nlp_project
    ```
2. **Set up a Python virtual environment (Recommended):**
    ```bash
   pip install pipenv
   pipenv shell
   ```
3. **Install dependencies:**
   ```bash
   pipenv install
   ```
4. **Environment Variables:**
Ensure you have the necessary environment variables set up or configure them in a .env file in the root directory.


## Running the project
### Training the Model
To train the model, execute the training script:
   ```bash
   python executor/train_executor.py 
   ```
### Running the Flask Application
1. Start the Flask app:
   ```bash
   python -m app.app
   ```
2. Access the application via http://localhost:5000 on your browser to interact with the API. 

### Testing the API
You can test the API by sending a POST request with some sample text. Here is how you can do it using curl:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"hobbies include fighting Mariam bitch"}'
   ```

### Docker Container 
To build and run the Docker container:

1. **Build the Docker image:**
   ```bash
   docker build -t nlp_project:latest .
   ```
2. **Run the container:**
   ```bash
   docker run -p 5000:5000 nlp_project:latest
   ```
### Logging
Logs are generated under the logs/ directory and can be reviewed for detailed application behavior and errors.

