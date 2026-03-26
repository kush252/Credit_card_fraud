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

def monitoring_logs(data_drift_decision:dict,concept_drift_results:dict, run_id:str):
    try:
        response = supabase.table("monitoring_logs").insert({
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "data_drift_detected": data_drift_decision['decision'] == "retrain",
            "concept_drift_detected": concept_drift_results['decision'] == "retrain",
            "retraining_triggered": data_drift_decision['decision'] == "retrain" or concept_drift_results['decision'] == "retrain"
        }).execute()
    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")

    return response
