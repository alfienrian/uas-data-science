from preprocessing import load_data, prepare_features
from train_baseline import train_baseline
from train_rf import train_rf
from train_dl import build_mlp, train_mlp, save_dl_model
from save_model import save_sklearn_model
from sklearn.model_selection import train_test_split

df = load_data("data/data.csv")
X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline
baseline = train_baseline(preprocessor, X_train, y_train)
save_sklearn_model(baseline, "models/model_baseline.pkl")

# Random Forest
rf = train_rf(preprocessor, X_train, y_train)
save_sklearn_model(rf, "models/model_rf.pkl")

# Deep Learning
X_train_dl = preprocessor.fit_transform(X_train)
input_dim = X_train_dl.shape[1]
num_classes = len(set(y))

dl_model = build_mlp(input_dim, num_classes)
dl_model, history, train_time = train_mlp(dl_model, X_train_dl, y_train)
save_dl_model(dl_model, "models/model_mlp.h5")
