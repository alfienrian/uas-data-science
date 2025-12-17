from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from save_model import save_sklearn_model

def train_rf(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model

# save_sklearn_model(model, "models/model_rf.pkl")
