import pandas as pd
from deployment.monitoring.drifts.data_drift import detect_data_drift, drift_decision
from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

def detect_concept_drift():
    df = fetch_recent_data()

    # only rows where label is available
    df = df.dropna(subset=["actual_label"])

    y_true = df["actual_label"]
    y_pred = df["prediction"]

    roc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    confusion_mat = confusion_matrix(y_true, y_pred)
    if f1 < 0.60:
        return {
            "metrics_before": {
                "roc_auc": roc,
                "f1_score": f1,
                "pr_auc": pr_auc,
                "confusion_matrix": confusion_mat.tolist()
            },
            "decision": "retrain"
        }
    else:
        return {
            "metrics_before": {
                "roc_auc": roc,
                "f1_score": f1,
                "pr_auc": pr_auc,
                "confusion_matrix": confusion_mat.tolist()
            },
            "decision": "stable"
        }

if __name__ == "__main__":
    concept_drift_results = detect_concept_drift()
    print("\nFinal Decision:", concept_drift_results["decision"])