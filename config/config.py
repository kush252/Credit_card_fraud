import os
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH =os.path.join(PROJECT_ROOT, "data\creditcard_scaled.csv")
DATA_METADATA_PATH = os.path.join(PROJECT_ROOT, "data\creditcard_scaled_metadata.json")
MODEL_METADATA_PATH = os.path.join(PROJECT_ROOT, "models\\final_model_metadata.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_config():
    config={
        "model_folder": MODELS_DIR,
        "data_path": DATA_PATH,
        "data_metadata_path": DATA_METADATA_PATH,
        "model_name": f"final_model_v{str(MODEL_VERSION)}",
        "model_version": MODEL_VERSION,
        "model_metadata_path": MODEL_METADATA_PATH
    }
    return config


if __name__ == "__main__":
    config = get_config()
    print(config)