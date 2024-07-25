import yaml
import os

def load_config():
    # Get the directory of the current file (config.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to config.yaml
    config_path = os.path.join(current_dir, 'config.yaml')

    # Open and load the config file
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)