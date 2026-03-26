from deployment.monitoring.utils.logging import get_logs_count
from deployment.monitoring.monitoring import run_monitoring
from deployment.retraining.retrain_model import retrain_model
import uuid


def monitoring_retraining_pipeline():
    logs_count = get_logs_count()
    if logs_count <= 10:
        return 
    
    run_id = str(uuid.uuid4())
    decision, concept_drift_results, data_drift_score = "stable", {}, 0
    
    decision, concept_drift_results, data_drift_score = run_monitoring(run_id)
    metrics_before = concept_drift_results.get("metrics_before", {})
    if decision == "retrain":
        retrain_model(metrics_before, data_drift_score,run_id)


if __name__ == "__main__":
    monitoring_retraining_pipeline()