from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix
from src.utils.model_load_save import save_model
from sklearn.ensemble import RandomForestClassifier

import joblib
import os
import json
from dotenv import load_dotenv
import uuid

from deployment.retraining.fetch_retraining_data import load_retraining_data
from config.config import get_config
from src.utils.model_creator import create_basic_model
from deployment.logging.retraining_logs import retraining_logs
from deployment.logging.system_events_logs import system_events_logs


load_dotenv()

def increment_model_version(env_path=".env"):

    current_version = int(os.getenv("MODEL_VERSION", 1))
    new_version = current_version + 1

    with open(env_path, "r") as f:
        lines = f.readlines()

    with open(env_path, "w") as f:
        found = False
        for line in lines:
            if line.startswith("MODEL_VERSION="):
                f.write(f"MODEL_VERSION={new_version}\n")
                found = True
            else:
                f.write(line)

        if not found:
            f.write(f"\nMODEL_VERSION={new_version}\n")

    return new_version


def retrain_model(metrics_before:dict, data_drift_score:float,run_id=None):
    config = get_config()
    current_version = int(os.getenv("MODEL_VERSION", 1))
    version = current_version + 1

    if not run_id:
        run_id = str(uuid.uuid4())
    data = load_retraining_data()

    X = data.drop(columns=["Class"])
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dataset_size = len(data)
    retraining_triggered = False
    status = "retraining_started"
    event_type = "retraining"
    message = "retraining process initiated. "
    system_events_logs({
        "event_type": event_type,
        "status": status,
        "message": message
    }, run_id)

    try:
        past_model_metadata = json.load(open(config["model_metadata_path"]))
        model_name = past_model_metadata.get("model_name", "random_forest")
        hyperparameters = past_model_metadata.get("hyperparameters", {})
    except Exception as e:
        model_name = "random_forest"
        hyperparameters = {}
        status  = f"failed_to_load_past_metadata: {str(e)}"
        message = f"Past model metadata failed to load due to error: {str(e)}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
    
    model = create_basic_model(model_name)
    model.set_params(**hyperparameters)

    try:
        model.fit(X_train, y_train)
        retraining_triggered = True
        status = "model_retrained"
        message = "Model retrained successfully."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
    except Exception as e:
        status = f"retraining_failed: {str(e)}"
        message = f"Model retraining failed due to error: {str(e)}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        return None, None

    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    roc_score = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, predicted)
    pr_score = average_precision_score(y_test, proba)
    

    confusion = confusion_matrix(y_test, predicted)

    metrics_after = {
        "roc_auc": roc_score,
        "f1_score": f1,
        "pr_auc": pr_score,
        "confusion_matrix": confusion.tolist()
    }

    metadata = {
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "test_metrics": {
            "roc_auc": roc_score,
            "f1_score": f1,
            "pr_auc": pr_score,
            "confusion_matrix": confusion.tolist()
        }
    }
    try:
        save_model(model, metadata, "final_model_v" + str(version))
        new_version = increment_model_version(".env")
        status = "model_saved and new version implemented"
        message = "Model saved successfully with new version."
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
    except Exception as e:
        status = f"model_saving_failed: {str(e)}"
        message = f"Model saving failed due to error: {str(e)}"
        system_events_logs({
            "event_type": event_type,
            "status": status,
            "message": message
        }, run_id)
        return None, None




    retraining_logs(dataset_size,retraining_triggered, status, metrics_after,metrics_before,new_version,run_id)
    message = f"Retraining completed. New model version: {new_version}. New model metrics: {metrics_after}"
    data_for_system_events_logs = {
        "event_type": event_type,
        "status": status,
        "message": message
    }
    system_events_logs(data_for_system_events_logs, run_id)
    return model,metadata



if __name__ == "__main__":
    model, metadata = retrain_model() # will give error args missing coz bored
    print("Retrained model metadata:", metadata)
    