import os
import joblib

# Detect project root automatically (2 folders above /web/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_model():
    """Load trained model and scaler from /models/."""
    print("🔍 Looking for model in:", MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"❌ Scaler not found at {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully from:", MODEL_DIR)
    return model, scaler

def save_model(model, scaler):
    """Save model and scaler to /models/."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model and Scaler saved successfully to:", MODEL_DIR)
