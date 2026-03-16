import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.dataloader import load_data
from src.utils.traintestsplit import train_val_test_split
from src.utils.model_load_save import save_model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score,f1_score,confusion_matrix,average_precision_score

import joblib
import json

from config.config import get_config

config = get_config()
data_path = config["data_path"]

X, y = load_data(data_path)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_val)[:, 1]
predictions = model.predict(X_val)

roc_score = roc_auc_score(y_val, probs)
f1 = f1_score(y_val, predictions)

print(f"Baseline Logistic Regression -> ROC-AUC: {roc_score:.4f} | F1: {f1:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_val, predictions))
print("Precision-Recall curve:\n", average_precision_score(y_val, probs))





metadata = {
    "model_name": "Logistic Regression",
    "hyperparameters": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    },
    "validation_metrics": {
        "roc_auc": roc_score,
        "f1_score": f1
    }
}

save_model(model, metadata, "baseline_model")