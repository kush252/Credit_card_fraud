from config.config import get_config
import os
import requests
from deployment.logging.system_events_logs import system_events_logs 
import uuid

config = get_config()
MODEL_VERSION = config["model_version"]
MODEL_URL = str(os.getenv("MODEL_URL")) 
DATA_URL = str(os.getenv("DATA_URL", MODEL_URL))
MODEL_PATH = str(config["model_folder"] + "/")
DATA_PATH = os.path.dirname(config["data_path"])

def download_model(run_id=None):
    """Download model and model metadata from remote storage"""
    model_path = MODEL_PATH + f"final_model_v{MODEL_VERSION}.joblib"
    model_url = MODEL_URL + f"final_model_v{MODEL_VERSION}.joblib"
    model_metadata_path = config["model_metadata_path"]
    model_metadata_url = MODEL_URL + f"final_model_v{MODEL_VERSION}_metadata.json"
    
    if not run_id:
        run_id = str(uuid.uuid4())
    
    event_type = "model_download"
    status = "initiated"
    message = f"Model download initiated for version {MODEL_VERSION}. Model URL: {model_url}. Model Metadata URL: {model_metadata_url}. Model Path: {model_path}. Model Metadata Path: {model_metadata_path}."
    system_events_logs({
        "event_type": event_type,
        "status": status,
        "message": message
    }, run_id)

    if os.path.exists(model_path) and os.path.exists(model_metadata_path):
        status = "skipped"
        message = f"Model files already exist for version {MODEL_VERSION}. Skipping download."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
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
        message = f"Model download completed for version {MODEL_VERSION}. Model saved to: {model_path}. Metadata saved to: {model_metadata_path}."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
    except Exception as e:
        status = "failed"
        message = f"Model download failed for version {MODEL_VERSION}. Error: {str(e)}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        raise e


def download_data_files(run_id=None):
    """Download data files (CSV and metadata) from remote storage"""
    if not run_id:
        run_id = str(uuid.uuid4())
    
    data_files = [
        # ("creditcard_scaled.csv", "data_csv"),
        ("creditcard_scaled_metadata.json", "metadata_json"),
        ("data_distribution.json", "distribution_json")
    ]
    
    
    for filename, file_type in data_files:
        file_path = os.path.join(DATA_PATH, filename)
        file_url = DATA_URL + filename
        
        event_type = "data_download"
        status = "initiated"
        message = f"Data file download initiated. File: {filename}. URL: {file_url}. Local path: {file_path}."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        
        if os.path.exists(file_path):
            status = "skipped"
            message = f"Data file already exists: {filename}. Skipping download."
            system_events_logs({
                "event_type": event_type,
                "status": status,
                "message": message
            }, run_id)
            continue
        
        try:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            status = "completed"
            message = f"Data file download completed. File: {filename}. Size: {file_size_mb:.2f} MB. Saved to: {file_path}."
            system_events_logs({
                "event_type": event_type,
                "status": status,
                "message": message
            }, run_id)
            
        except requests.exceptions.Timeout:
            status = "failed"
            message = f"Data file download timed out. File: {filename}. URL: {file_url}."
            system_events_logs({
                "event_type": event_type,
                "status": status,
                "message": message
            }, run_id)
            raise
            
        except Exception as e:
            status = "failed"
            message = f"Data file download failed. File: {filename}. Error: {str(e)}"
            system_events_logs({
                "event_type": event_type,
                "status": status,
                "message": message
            }, run_id)
            raise e


def install_all(run_id=None):
    """Download all required files: model and data"""
    if not run_id:
        run_id = str(uuid.uuid4())
    
    event_type = "installation"
    status = "initiated"
    message = "Starting model and data downloads."
    system_events_logs({
        "event_type": event_type,
        "status": status,
        "message": message
    }, run_id)
    
    try:
        download_model(run_id)
        # download_data_files(run_id)
        
        status = "completed"
        message = "model downloaded."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        
    except Exception as e:
        status = "failed"
        message = f"Files installation failed. Error: {str(e)}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        raise e


if __name__ == "__main__":
    install_all()