import os
from supabase import create_client
from dotenv import load_dotenv
from config.config import get_config
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
config = get_config()
MODEL_VERSION = config["model_version"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_prediction(data: dict, prediction: int, probability: float):
    try:
        row = {
            **{f"v{i}": data.get(f"V{i}") for i in range(1, 29)},
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": MODEL_VERSION,
            "scaled_amount": data.get("scaled_amount"),
            "scaled_time": data.get("scaled_time"),
        }

        response = supabase.table("prediction_logs").insert(row).execute()

    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")