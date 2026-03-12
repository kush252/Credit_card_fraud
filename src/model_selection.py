from utils.dataloader import load_data
from utils.traintestsplit import train_val_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import numpy as np

def select_best_model(X_train, y_train, X_val, y_val):

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(), 
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():

        if name == "gradient_boosting":
            sample_weights = np.array([class_weight_dict[label] for label in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:,1]

        roc = roc_auc_score(y_val, probs)
        f1 = f1_score(y_val, preds)

        results[name] = {
            "roc_auc": roc,
            "f1": f1
        }
        fitted_models[name] = model

        print(f"{name} -> ROC-AUC: {roc:.4f} | F1: {f1:.4f}")

    best_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_model = fitted_models[best_name]

    return best_name, best_model, results