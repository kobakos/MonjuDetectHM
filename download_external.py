from cryoet_data_portal import Client, Dataset

# Instantiate a client, using the data portal GraphQL API by default
client = Client()

from pathlib import Path
from src.utils import load_configs
# Load the configuration
settings = load_configs('SETTINGS.json')
BASE_PATH = Path(settings['base_path'])
dataset = Dataset.get_by_id(client, 10441)
dataset.download_everything(dest_path=BASE_PATH / settings["external_data_path"])