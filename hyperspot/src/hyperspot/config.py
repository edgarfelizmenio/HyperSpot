import configparser

from enum import Enum
from pathlib import Path

# TODO:
# apply DDP to parallelize operations
# https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
# 
class AcceleratorOptions(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

class ClassifierMode(str, Enum):
    TRAIN = "train"
    ATTR = "attr"
    

# appdirs, configparser
config_path = Path(__file__).parent / 'settings.ini'

config_files = [
    config_path,
    'hyperspot_custom_settings.ini'
]

app_config = configparser.ConfigParser()
app_config.read(config_files)

project_files_dir = Path(app_config.get('cli', 'ProjectFilesDir'))