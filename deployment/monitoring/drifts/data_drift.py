import pandas as pd
from scipy.stats import ks_2samp
from supabase import create_client
import os
from dotenv import load_dotenv

from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_reference_data():
    """Fetch original training dataset from Supabase"""
    try:
        response = supabase.table("training_data").select("*").execute()
        reference_df = pd.DataFrame(response.data)
        return reference_df
    except Exception as e:
        print(f"Error loading reference data from database: {str(e)}")
        raise e


def detect_data_drift():
    recent_data = fetch_recent_data()
    reference_data = load_reference_data()
    
    features = [col for col in reference_data.columns if col != "Class"]

    drift_results = {}

    for feature in features:
        reference_values = reference_data[feature].dropna().tolist()
        
        # safely find column name in recent_data
        recent_col = feature.lower()
        if recent_col == "time" and "scaled_time" in recent_data.columns:
            recent_col = "scaled_time"
        elif recent_col == "amount" and "scaled_amount" in recent_data.columns:
            recent_col = "scaled_amount"
            
        if recent_col not in recent_data.columns:
            continue
            
        current_values = recent_data[recent_col].dropna().tolist()

        if len(current_values) < 5:
            continue

        stat, p_value = ks_2samp(reference_values, current_values)

        drift = p_value < 0.05

        drift_results[feature] = {
            "p_value": p_value,
            "drift": drift
        }

    return drift_results


def drift_decision(drift_results):
    
    total_features = len(drift_results)
    drifted_features = sum(1 for f in drift_results.values() if f["drift"])

    drift_score = drifted_features / total_features if total_features > 0 else 0

    # decision rule
    if drift_score > 0.3:
        decision = "retrain"
    else:
        decision = "stable"

    return {
        "drift_score": drift_score,
        "decision": decision
    }


if __name__ == "__main__":
    drift_results = detect_data_drift()
    decision = drift_decision(drift_results)

    print("\nFinal Decision:", decision)