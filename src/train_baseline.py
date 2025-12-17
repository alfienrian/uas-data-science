from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from save_model import save_sklearn_model

def train_baseline(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model

# Example usage
# model = train_baseline(...)
# save_sklearn_model(model, "models/model_baseline.pkl")
