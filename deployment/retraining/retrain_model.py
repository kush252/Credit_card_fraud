from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix
from src.utils.model_load_save import save_model
from sklearn.ensemble import RandomForestClassifier

import joblib
import os
import json
from dotenv import load_dotenv

from deployment.retraining.fetch_retraining_data import load_retraining_data
from config.config import get_config
from src.utils.model_creator import create_basic_model



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

def retrain_model():
    config = get_config()
    version = config["model_version"] + 1
    print(f"Retraining model to version v{version}...")
    
    data = load_retraining_data()

    X = data.drop(columns=["Class"])
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    past_model_metadata = json.load(open(config["model_metadata_path"]))
    model_name = past_model_metadata.get("model_name", "random_forest")
    hyperparameters = past_model_metadata.get("hyperparameters", {})
    
    model = create_basic_model(model_name)
    model.set_params(**hyperparameters)


    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    roc_score = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, predicted)
    pr_score = average_precision_score(y_test, proba)

    print(f"Final {model_name} -> ROC-AUC: {roc_score:.4f} | F1: {f1:.4f} | PR-AUC: {pr_score:.4f}")
    confusion = confusion_matrix(y_test, predicted)
    print("Confusion Matrix:\n", confusion)

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

    save_model(model, metadata, "final_model_v" + str(version))
    new_version = increment_model_version(".env")

    return model,metadata



if __name__ == "__main__":
    model, metadata = retrain_model()
    print("Retrained model metadata:", metadata)
    