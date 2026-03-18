import os
from supabase import create_client
import supabase


from dotenv import load_dotenv
from config.config import get_config
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
config = get_config()
MODEL_VERSION = config["model_version"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def update_label(row_id, actual_label):
    supabase.table("prediction_logs") \
        .update({"actual_label": actual_label}) \
        .eq("id", row_id) \
        .execute()