import os
import joblib
import pandas as pd

# === Model Paths ===
BASE_DIR = r"C:\Users\AVADUT\RealTime_Network_Analyzer"
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def load_model():
    """Load saved RandomForest model and scaler safely."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
    return model, scaler


def predict_from_features(model, scaler, features):
    """
    Ensure consistent DataFrame structure before prediction.
    Accepts a list of lists or DataFrame.
    """
    if not isinstance(features, pd.DataFrame):
        df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(1, len(features[0]) + 1)])
    else:
        df = features.copy()

    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    return preds
