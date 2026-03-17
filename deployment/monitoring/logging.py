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
            "amount": data.get("scaled_amount"),
            "time": data.get("scaled_time"),
            **{f"v{i}": data.get(f"V{i}") for i in range(1, 29)},
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": MODEL_VERSION,
        }

        response = supabase.table("prediction_logs").insert(row).execute()

    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")