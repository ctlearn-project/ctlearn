from ctlearn.run_model import run_model
from tensorflow.keras import models
import yaml
import os


def train_predictor(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # TODO: os.makedirs should be unnecessary
    os.makedirs(config['Logging']['model_directory'], exist_ok=True)
    run_model(config, mode='train')

    return config['Logging']['model_directory']


def get_predictor(predefined_model_path, config_path):
    # Load the predictor if a predefined model is specified and it exists
    if predefined_model_path and os.path.exists(predefined_model_path):
        predictor = models.load_model(predefined_model_path)
    # Otherwise, train a new predictor
    else:
        predictor_path = train_predictor(config_path)
        predictor = models.load_model(predictor_path)

    return predictor