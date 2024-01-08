from appdirs import user_data_dir
from pathlib import Path
import toml
import os
from mlroom.utils import mlutils as mu

#DEF_TOML_FILE = 'config.toml'
DEF_TOML_FILE = 'config_tick.toml'
DEF_CACHE_FILE = 'training_data.joblib'

def load_config(filename):
    filename = filename if filename is not None else DEF_TOML_FILE
    # Path to the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the .cfg file
    cfg_file = os.path.join(script_dir, filename)

    with open(cfg_file, 'r') as file:
        config_string = file.read()
    config = toml.loads(config_string)
    print(mu.red(f"CONFIG {filename} LOADED"))
    return config, config_string

# # Path to the directory containing the script
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # Path to the .cfg file
# cfg_file = os.path.join(script_dir, TOML_FILE)
# CONFIG, CONFIG_STRING = load_config(cfg_file)

#urcuje dostupne zdroje a jejich granularitu, od nejvyssi po nejnizsi - pouziva se pri sekvencingu
SOURCES_GRANULARITY = dict(cbar_indicators=10, bars=9, indicators=9, dailyBars=5)

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
WEB_API_KEY="david"
DATA_DIR = user_data_dir("models")