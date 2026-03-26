import pandas as pd
import json
from scipy.stats import ks_2samp

from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data


def detect_data_drift():
    recent_data = fetch_recent_data()

    with open("data/creditcard_scaled_metadata.json") as f:
        metadata = json.load(f)

    feature_stats = metadata["feature_stats"]
    features = list(feature_stats.keys())

    drift_results = {}

    for feature in features:
        reference_values = feature_stats[feature]["values"]
        current_values = recent_data[feature.lower()].dropna().tolist()

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