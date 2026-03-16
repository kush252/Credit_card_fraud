import os

from src.workers.model_selection import select_best_model
from src.utils.dataloader import load_data
from src.utils.traintestsplit import train_val_test_split
from src.workers.hyperparameter import tune_model
from src.utils.model_load_save import save_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,f1_score,average_precision_score,confusion_matrix
from xgboost import XGBClassifier
import numpy as np
from config.config import get_config


config = get_config()
data_path = config["data_path"]


X, y = load_data(data_path)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)



best_name, results = select_best_model(
        X_train, y_train, X_val, y_val
    )

best_name, best_params = tune_model(best_name, X_train, y_train, X_val, y_val)

X_train_final = np.concatenate([X_train, X_val])
y_train_final = np.concatenate([y_train, y_val])

print("Best model after tuning:", best_name)
print("Best parameters after tuning:", best_params)


def build_model(name, params):

    if name == "random_forest":
        return RandomForestClassifier(**params)

    elif name == "gradient_boosting":
        return GradientBoostingClassifier(**params)

    elif name == "logistic_regression":
        return LogisticRegression(**params)

    elif name == "xgboost":
        return XGBClassifier(**params)
    

final_model = build_model(best_name, best_params)

final_model.fit(X_train_final, y_train_final)
predicted = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, predicted)
pr_score = average_precision_score(y_test, proba)

print(f"Final {best_name} -> ROC-AUC: {roc_score:.4f} | F1: {f1:.4f} | PR-AUC: {pr_score:.4f}")
confusion = confusion_matrix(y_test, predicted)
print("Confusion Matrix:\n", confusion)

metadata = {
    "model_name": best_name,
    "hyperparameters": best_params,
    "test_metrics": {
        "roc_auc": roc_score,
        "f1_score": f1,
        "pr_auc": pr_score,
        "confusion_matrix": confusion.tolist()
    }
}

save_model(final_model, metadata, "final_model")

save_model(final_model, metadata, "final_model")

