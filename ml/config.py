from appdirs import user_data_dir
from pathlib import Path
import toml
import os

def load_config(filename):
    with open(filename, 'r') as file:
        config = toml.load(file)
    return config

# Path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the .cfg file
cfg_file = os.path.join(script_dir, 'config.toml')
CONFIG = load_config(cfg_file)["configuration"]
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
WEB_API_KEY="david"
DATA_DIR = user_data_dir("models")