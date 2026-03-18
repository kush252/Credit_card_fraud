from supabase import create_client
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def fetch_recent_data(limit=1000):
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        supabase = create_client(supabase_url, supabase_key)
        
        response = supabase.table("prediction_logs") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()

        return pd.DataFrame(response.data)
    
    except Exception as e:
        print(f"[FETCH ERROR]: Failed to fetch recent data - {e}")
        raise