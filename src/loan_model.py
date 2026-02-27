from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, mae, r2

def save_model(model, filepath="loan_model.pkl"):
    joblib.dump(model, filepath)

def load_model(filepath="loan_model.pkl"):
    return joblib.load(filepath)