import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

print("Connected to Supabase!")

data = {
    "amount": 100.0,
    "time": 50.0,

    **{f"v{i}": 0.1 for i in range(1, 29)},

    "prediction": 0,
    "probability": 0.23,
    "model_version": "v1"
}

response = supabase.table("prediction_logs").insert(data).execute()

print(response)