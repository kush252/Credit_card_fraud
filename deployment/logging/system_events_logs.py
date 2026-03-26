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

def system_events_logs(data, run_id):
    try:
        response = supabase.table("system_events").insert({
            "run_id": run_id,
            "event_type": data.get("event_type", "unknown"),
            "status": data.get("status", "unknown"),
            "message": data.get("message", ""),
        }).execute()
    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")

    return response
