import pandas as pd
import numpy as np
import json
import config.config as config

def simulation_data_generation():
    config_data = config.get_config()
    data_distribution_path = config_data["data_distribution_path"]
    data_distribution = {}
    with open(data_distribution_path, "r") as f:
        data_distribution = json.load(f)
    
    pca_columns = data_distribution["pca"].keys()
    non_pca_columns = data_distribution["non_pca"].keys()
    values = {}
    for col in pca_columns:
        mean = data_distribution["pca"][col]["mean"]
        std = data_distribution["pca"][col]["std"]
        values[col] = np.random.normal(loc=mean, scale=std, size=50)

    for col in non_pca_columns:
        mean = data_distribution["non_pca"][col]["mean"]
        std = data_distribution["non_pca"][col]["std"]
        values[col] = np.random.normal(loc=mean, scale=std, size=50)

    simulated_df = pd.DataFrame(values)
    return simulated_df


if __name__ == "__main__":
    simulated_df = simulation_data_generation()
    print(simulated_df.head())