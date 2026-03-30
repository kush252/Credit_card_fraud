import pandas as pd
from deployment.monitoring.drifts.data_drift import detect_data_drift, drift_decision
from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def detect_concept_drift():
    df = fetch_recent_data()

    if 'actual_label' in df.columns:
        df = df.dropna(subset=["actual_label"])
    else:
        df = pd.DataFrame() 

    if len(df) < 2:
        logger.info("Not enough actual_labels to calculate concept drift. Defaulting to stable.")
        return {
            "metrics_before": {
                "roc_auc": 0,
                "f1_score": 0,
                "pr_auc": 0,
                "confusion_matrix": [[0,0],[0,0]]
            },
            "decision": "stable"
        }
        
    y_true = df["actual_label"]
    y_pred = df["prediction"]

    if len(y_true.unique()) < 2:
        logger.warning(f"Only one class present in actual_labels (class: {y_true.unique()[0]}). Metrics might be inaccurate.")
        roc = 0.5 #
    else:
        roc = roc_auc_score(y_true, y_pred)
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Try PR AUC; if fails due to single class, fallback
    try:
        pr_auc = average_precision_score(y_true, y_pred)
    except Exception:
        pr_auc = 0.0
        
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