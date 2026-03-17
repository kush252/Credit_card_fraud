from deployment.monitoring.logging import log_prediction

sample_data = {
    "amount": 250.0,
    "time": 100.0,
    **{f"v{i}": 0.5 for i in range(1, 29)}
}

log_prediction(sample_data, prediction=1, probability=0.87)

print("Logged successfully!")