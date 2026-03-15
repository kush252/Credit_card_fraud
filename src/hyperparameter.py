import optuna

from src.model_selection import select_best_model
from src.utils.dataloader import load_data
from src.utils.traintestsplit import train_val_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,f1_score,average_precision_score
from xgboost import XGBClassifier


def objective(trial, best_name, X_train, y_train, X_val, y_val):
    if best_name == "random_forest":

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "class_weight": "balanced",
            "random_state": 42
        }

        model = RandomForestClassifier(**params)


    elif best_name == "gradient_boosting":

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": 42
        }

        model = GradientBoostingClassifier(**params)


    elif best_name == "logistic_regression":

        params = {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "max_iter": trial.suggest_int("max_iter", 500, 2000),
            "class_weight": "balanced",
            "random_state": 42
        }

        model = LogisticRegression(**params)


    elif best_name == "xgboost":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42
        }

        model = XGBClassifier(**params)

    else:
        raise ValueError("Unknown model")

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probs)
    f1 = f1_score(y_val, model.predict(X_val))
    pr_auc = average_precision_score(y_val, probs)
    score = 0.4 * roc_auc + 0.6 * f1
    
    return pr_auc


def tune_model(best_name=None, X_train=None, y_train=None, X_val=None, y_val=None):

    #adding validation left
    
    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, best_name, X_train, y_train, X_val, y_val),
        n_trials=6
    )

    print("Best Average Precision-Recall score:", study.best_value)
    print("Best parameters:", study.best_params)

    return best_name, study.best_params


if __name__ == "__main__":
    X, y = load_data(r"D:\Kush\2nd Year\projects\MLOPs\data\creditcard_scaled.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    best_name, results = select_best_model(
        X_train, y_train, X_val, y_val
    )

    print("Best baseline model:", best_name)
    best_name, best_params = tune_model(best_name, X_train, y_train, X_val, y_val)