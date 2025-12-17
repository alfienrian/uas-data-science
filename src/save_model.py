import joblib
import os

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_sklearn_model(path):
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
