import requests
import time
import json
from deployment.testing.traffic_simulation.data_generation import simulation_data_generation

API_URL = "http://127.0.0.1:8000/predict"


def send_request(sample):
    try:
        response = requests.post(API_URL, json=sample, timeout=30)

        return {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else None
        }

    except Exception as e:
        return {
            "status_code": "error",
            "error": str(e)
        }


def run_simulation(delay=5):
    logs = []
    sample_df = simulation_data_generation()
    
    for i in range(1):
        sample = sample_df.iloc[i].to_dict()
        print(f"Sending request {i+1}/{len(sample_df)}: {sample}")
        result = send_request(sample)


    return result


if __name__ == "__main__":
    result = run_simulation()
    print("Simulation completed. Logs:")
    print(json.dumps(result, indent=4))