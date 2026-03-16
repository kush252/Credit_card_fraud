import os
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_config():
    config={
        "model_folder": MODELS_DIR,
        "data_path": r"D:\Kush\2nd Year\projects\MLOPs\data\creditcard_scaled.csv",
        "data_metadata_path": r"D:\Kush\2nd Year\projects\MLOPs\data\creditcard_scaled_metadata.json",
        "model_name": "baseline_model"
    }
    return config