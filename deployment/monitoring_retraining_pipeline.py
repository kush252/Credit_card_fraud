from deployment.monitoring.utils.logging import get_logs_count
from deployment.monitoring.monitoring import run_monitoring
from deployment.retraining.retrain_model import retrain_model



def monitoring_retraining_pipeline():
    logs_count = get_logs_count()
    if logs_count <= 10:
        return 
    
    decision = "stable"
    decision = run_monitoring()

    if decision == "retrain":
        retrain_model()


if __name__ == "__main__":
    monitoring_retraining_pipeline()