import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def analyze_data_distribution(df):
    pca_columns = [col for col in df.columns if col.startswith("V") and col != "Class"]
    non_pca_columns = [col for col in df.columns if not col.startswith("V") and col != "Class"]

    pca_stats_df = df[pca_columns].describe().T
    non_pca_stats_df = df[non_pca_columns].describe().T

    pca_column_stats = pca_stats_df.to_dict(orient="index")
    non_pca_column_stats = non_pca_stats_df.to_dict(orient="index")

    with open("data/data_distribution.json", "w") as f:
        json.dump({"pca": pca_column_stats, "non_pca": non_pca_column_stats}, f)

    return pca_column_stats, non_pca_column_stats



if __name__ == "__main__":

    df = pd.read_csv("data/creditcard_scaled.csv")

    pca_stats, non_pca_stats = analyze_data_distribution(df)

