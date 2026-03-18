from deployment.monitoring.drifts.data_drift import detect_data_drift, drift_decision
from deployment.monitoring.drifts.concept_drift import detect_concept_drift

def run_monitoring():
    print("Running Data Drift Detection...")
    data_drift_results = detect_data_drift()
    data_drift_decision = drift_decision(data_drift_results)

    print("\nRunning Concept Drift Detection...")
    concept_drift_results = detect_concept_drift()

    print("\n=== Monitoring Summary ===")
    print(f"Data Drift Decision: {data_drift_decision['decision']} (Score: {data_drift_decision['drift_score']:.2f})")
    print(f"Concept Drift Decision: {concept_drift_results['decision']} (Recall: {concept_drift_results['recall']:.3f})")

    # Overall decision
    if data_drift_decision['decision'] == "retrain" or concept_drift_results['decision'] == "retrain":
        print("\n⚠️  OVERALL DECISION: RETRAIN MODEL")
        return "retrain"
    else:
        print("\n✅ OVERALL DECISION: MODEL STABLE")
        return "stable"


if __name__ == "__main__":
    decision = run_monitoring()