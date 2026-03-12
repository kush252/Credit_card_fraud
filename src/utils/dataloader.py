import pandas as pd

def data_validator(df):

    required_cols = ["Time", "Amount", "Class"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Null values detected")

    if df.duplicated().sum() > 0:
        raise ValueError("Duplicate rows detected")
    

def load_data(file_path):
    df= pd.read_csv(file_path)
    data_validator(df)
    print("Data Loaded Successfully")
    print("Shape:", df.shape)
    print("Fraud Cases:", df["Class"].sum())
    return df