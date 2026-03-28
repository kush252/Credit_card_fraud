import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv
from config.config import get_config
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
config = get_config()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

df = pd.read_csv(config["data_path"])

df.columns = [col.lower() for col in df.columns]

batch_size = 500

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].to_dict(orient="records")
    supabase.table("training_data").insert(batch).execute()

print("Upload complete")