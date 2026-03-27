from config.config import get_config
import os
import requests
from deployment.logging.system_events_logs import system_events_logs 
import uuid

config = get_config()
MODEL_VERSION = config["model_version"]
MODEL_URL = os.getenv("MODEL_URL") 
MODEL_PATH = config["model_folder"] 

def download_model(run_id=None):
    model_path = MODEL_PATH + f"final_model_v{MODEL_VERSION}.joblib"
    model_url = MODEL_URL + f"_final_model_v{MODEL_VERSION}.joblib"
    model_metadata_path = config["model_metadata_path"] + f"_v{MODEL_VERSION}.json"
    model_metadata_url = MODEL_URL + f"final_model_v{MODEL_VERSION}_metadata.json"
    if not run_id:
        run_id = str(uuid.uuid4())
    event_type = "model_download"
    status = "initiated"
    message = f"Model download initiated for version {MODEL_VERSION}.Model URL: {model_url}.Model Metadata URL: {model_metadata_url}.Model Path: {model_path}.Model Metadata Path: {model_metadata_path}."
    system_events_logs({
        "event_type": event_type,
        "status": status,
        "message": message
    }, run_id)

    if os.path.exists(model_path) and os.path.exists(model_metadata_path):
        return

    try:
        r = requests.get(model_url)
        rm = requests.get(model_metadata_url)
        r.raise_for_status()
        rm.raise_for_status()


        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        with open(model_path, "wb") as f:
            f.write(r.content)

        with open(model_metadata_path, "wb") as f:
            f.write(rm.content)

        status = "completed"
        message = f"Model download completed for version {MODEL_VERSION}."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
    except Exception as e:
        status = "failed"
        message = f"Model download failed for version {MODEL_VERSION}. Error: {e}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        raise e


if __name__ == "__main__":
    download_model()