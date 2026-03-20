from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

def create_basic_model(model_name: str):
    models = {
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(), 
        "xgboost": XGBClassifier()
    }
    return models.get(model_name)