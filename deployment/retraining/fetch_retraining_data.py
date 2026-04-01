import pandas as pd

from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data
from config.config import get_config

from supabase import create_client
import os
from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(level=logging.INFO)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
config = get_config()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def preprocess_logged_data(df):
    df = df.rename(columns={
        **{f"v{i}": f"V{i}" for i in range(1, 29)},
        'class': 'Class'
    })

    required_cols = (
        [f"V{i}" for i in range(1, 29)] +
        ["scaled_amount", "scaled_time", "Class"]
    )

    df = df[required_cols]

    return df

def load_original_data():
    try:
        response = supabase.table("training_data").select("*").execute()
        original_df = pd.DataFrame(response.data)
        logging.info(f"Loaded original data from database with columns: {original_df.columns.tolist()}")
        original_df = preprocess_logged_data(original_df)
        logging.info(f"Preprocessed original data columns: {original_df.columns.tolist()}")
        return original_df
        
    except Exception as e:
        logging.error(f"Error loading original data from database: {str(e)}")
        # Fallback to local CSV
        df = pd.read_csv(config["data_path"])
        return preprocess_logged_data(df)


def load_retraining_data():
    original = load_original_data()
    new_data = fetch_recent_data(limit=5000)
    new_data = new_data.dropna(subset=["actual_label"])
    new_data = new_data.rename(columns={"actual_label": "Class"})
    new_data = preprocess_logged_data(new_data)
    combined = pd.concat([original, new_data], ignore_index=True)

    return combined

if __name__ == "__main__":
    training_data = load_retraining_data()
    logging.info(f"Combined dataset shape: {training_data.shape}")
    logging.info(f"Combined dataset columns: {training_data.columns.tolist()}"  )