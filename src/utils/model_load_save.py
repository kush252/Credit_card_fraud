import os
import json
import joblib

BASE_DIR = os.path.dirname(__file__)
print("Base directory:", BASE_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
print("Project root:", PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
print("Models directory:", MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model, metadata: dict, model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    metadata_path = os.path.join(MODELS_DIR, f"{model_name}_metadata.json")

    joblib.dump(model, model_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


def load_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    metadata_path = os.path.join(MODELS_DIR, f"{model_name}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if "params" in metadata:
        try:
            model.set_params(**metadata["params"])
        except Exception:
            print("Warning: Could not set parameters from metadata. The model may not be configured correctly.")
            pass

    return model, metadata