import pandas as pd

from deployment.monitoring.utils.fetch_recent_data import fetch_recent_data
from config.config import get_config


def preprocess_logged_data(df):
    df = df.rename(columns={
        **{f"v{i}": f"V{i}" for i in range(1, 29)},
    })

    required_cols = (
        [f"V{i}" for i in range(1, 29)] +
        ["scaled_amount", "scaled_time", "Class"]
    )

    df = df[required_cols]

    return df
def load_retraining_data():
    config= get_config()
    original = pd.read_csv(config["data_path"])
    new_data = fetch_recent_data(limit=5000)
    new_data = new_data.dropna(subset=["actual_label"])
    new_data = new_data.rename(columns={"actual_label": "Class"})
    new_data = preprocess_logged_data(new_data)
    combined = pd.concat([original, new_data], ignore_index=True)

    return combined

if __name__ == "__main__":
    training_data = load_retraining_data()
    print(f"Combined dataset shape: {training_data.shape}")
    print(training_data.head())
    print("Columns:", training_data.columns.tolist())