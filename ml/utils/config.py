from appdirs import user_data_dir
from pathlib import Path
import toml
import os

def load_config(filename):
    with open(filename, 'r') as file:
        config = toml.load(file)
    return config

CONFIG = load_config("config.TOML")["configuration"]
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
WEB_API_KEY="david"
DATA_DIR = user_data_dir("models")