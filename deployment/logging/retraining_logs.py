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

def retraining_logs(dataset_size:int,retraining_triggered: bool, status: str, metrics_after: dict, metrics_before: dict,new_model_version:int, run_id:str):
    try:
        response = supabase.table("retraining_logs").insert({
            "run_id": run_id,
            "old_model_version": MODEL_VERSION,
            "new_model_version": new_model_version,
            "dataset_size": dataset_size,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "retraining_trigger": retraining_triggered,
            "status": status
        }).execute()

    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")

    return response
