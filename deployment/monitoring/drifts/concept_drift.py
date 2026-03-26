import pandas as pd
from deployment.monitoring.drifts.data_drift import detect_data_drift, drift_decision
from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data
from sklearn.metrics import precision_score, recall_score, f1_score




def detect_concept_drift():
    df = fetch_recent_data()

    # only rows where label is available
    df = df.dropna(subset=["actual_label"])

    y_true = df["actual_label"]
    y_pred = df["prediction"]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if f1 < 0.60:
        return {
            "f1": f1,
            "decision": "retrain"
        }
    else:
        return {
            "f1": f1,
            "decision": "stable"
        }

if __name__ == "__main__":
    concept_drift_results = detect_concept_drift()
    decision = drift_decision(concept_drift_results)

    print("\nFinal Decision:", decision)