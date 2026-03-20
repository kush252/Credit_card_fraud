from src.utils.dataloader import load_data
from src.utils.traintestsplit import train_val_test_split

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from config.config import get_config
from src.utils.model_creator import create_basic_model

def select_best_model(X_train, y_train, X_val, y_val):

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    models = {
        "logistic_regression": create_basic_model("logistic_regression"),
        "random_forest": create_basic_model("random_forest"),
        "gradient_boosting": create_basic_model("gradient_boosting"),
        "xgboost": create_basic_model("xgboost")
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        if name == "logistic_regression":
            model.set_params(
                max_iter=1000,
                class_weight="balanced"
            )

        elif name == "random_forest":
            model.set_params(
                class_weight="balanced"
            )

        elif name == "gradient_boosting":
            model.set_params()
            # (no extra params — matches your dict)

        elif name == "xgboost":
            model.set_params(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            )


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

    return best_name, results


if __name__ == "__main__":
    config = get_config()
    data_path = config["data_path"]
    X, y = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    best_name, results = select_best_model(X_train, y_train, X_val, y_val)
    print(f"Best Model: {best_name} with ROC-AUC: {results[best_name]['roc_auc']:.4f} and F1: {results[best_name]['f1']:.4f}")
