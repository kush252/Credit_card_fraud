from deployment.monitoring.drifts.data_drift import detect_data_drift, drift_decision
from deployment.monitoring.drifts.concept_drift import detect_concept_drift
from deployment.logging.monitoring_logs import monitoring_logs
from deployment.logging.system_events_logs import system_events_logs
import logging

from config.config import get_config    

logger = logging.getLogger(__name__)

def run_monitoring(run_id):
    config = get_config()
    event_status = "completed"
    try:
        logger.info("Detecting data drift...")
        data_drift_results = detect_data_drift()
        data_drift_decision = drift_decision(data_drift_results)
        logger.info(f"Data drift decision: {data_drift_decision['decision']} (score: {data_drift_decision['drift_score']})")

    except Exception as e:
        logger.error(f"Error in data drift detection: {e}", exc_info=True)
        data_drift_decision = {
        "drift_score": 0,
        "decision": "error"
    }

    try:
        logger.info("Detecting concept drift...")
        concept_drift_results = detect_concept_drift()
        logger.info(f"Concept drift decision: {concept_drift_results['decision']}")
    except Exception as e:
        logger.error(f"Error in concept drift detection: {e}", exc_info=True)
        concept_drift_results = {
            "metrics_before": {
                "roc_auc": 0,
                "f1_score": 0,
                "pr_auc": 0,
                "confusion_matrix": [[0,0],[0,0]]
            },
            "decision": "error"

        }

    if data_drift_decision['decision'] == "error" or concept_drift_results['decision'] == "error":
        event_status = "error"  
    else:
        logger.info("Logging monitoring results to database...")
        monitoring_logs(data_drift_decision,concept_drift_results,run_id)

    logger.info("Logging system events to database...")
    system_events_logs({
        "event_type": "monitoring_run",
        "status": event_status,
        "message": f"Data Drift Decision: {data_drift_decision['decision']}, Concept Drift Decision: {concept_drift_results['decision']}"
    },run_id)


    if data_drift_decision['decision'] == "retrain" or concept_drift_results['decision'] == "retrain":
        return "retrain",concept_drift_results,data_drift_decision['drift_score']
    else:
        return "stable",concept_drift_results,data_drift_decision['drift_score']

if __name__ == "__main__":
    decision = run_monitoring(run_id="test")