from deployment.monitoring.utils.logging import get_logs_count
from deployment.monitoring.monitoring import run_monitoring
from deployment.retraining.retrain_model import retrain_model
import uuid
import logging
import sys

# Configure logging at the INFO level so background tasks log to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def monitoring_retraining_pipeline():
    try :
        logger.info("Starting monitoring and retraining pipeline...")
        logs_count = get_logs_count()
        logger.info(f"Current prediction logs count: {logs_count}")
        if logs_count <= 5000:
            logger.info("Not enough logs (<=10) to run monitoring. Exiting pipeline.")
            return 
        
        run_id = str(uuid.uuid4())
        decision, concept_drift_results, data_drift_score = "stable", {}, 0
        
        logger.info(f"Running monitoring with run_id {run_id}...")
        decision, concept_drift_results, data_drift_score = run_monitoring(run_id)
        
        logger.info(f"Monitoring finished. Decision: {decision}, Data Drift Score: {data_drift_score}")
        metrics_before = concept_drift_results.get("metrics_before", {})
        if decision == "retrain":
            logger.info("Triggering retrain_model...")
            retrain_model(metrics_before, data_drift_score,run_id)
        else:
            logger.info("Condition is stable. No retraining triggered.")
    except Exception as e:
        logger.error(f"Error in monitoring and retraining pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    monitoring_retraining_pipeline()