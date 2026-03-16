from src.utils.model_load_save import load_model
import json

import pandas as pd
from config.config import get_config    

config = get_config()
MODEL_NAME = config["model_name"]
DATA_METADATA_PATH= config["data_metadata_path"]
def predict(inputs:dict):
    
    model, metadata = load_model(MODEL_NAME)
    data_metadata = json.load(open(DATA_METADATA_PATH))
    data_columns = data_metadata["columns"]
    filtered_data = {
        col: inputs.get(col, 0) 
        for col in data_columns 
        if col != "Class"
    }
    X = pd.DataFrame([filtered_data])
    
    params = metadata.get("hyperparameters", {})
    model.set_params(**params)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities     


if __name__ == "__main__":
    inputs = {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "scaled_amount": 1.7832739467616854,
        "scaled_time": -0.9949834936970594
    }
    predictions, probabilities = predict(inputs)
    print(predictions, probabilities)