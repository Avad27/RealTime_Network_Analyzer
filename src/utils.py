import os
import joblib
import numpy as np
import pandas as pd

# Set absolute project root to avoid ambiguity
PROJECT_ROOT = r"C:\Users\AVADUT\RealTime_Network_Analyzer"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_model():
    """Load model and scaler from models folder (raises clear error if missing)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # scaler must be a fitted sklearn scaler object
    return model, scaler

def align_and_scale_features(df_features, scaler):
    """
    Align incoming DataFrame (columns may be subset) to the scaler's expected input shape,
    filling missing columns with zeros and ordering correctly.
    Returns a numpy array scaled and ready for model.predict.
    """
    # If scaler has attribute 'mean_' it's fitted: determine expected feature count
    if not hasattr(scaler, "mean_"):
        raise ValueError("Scaler does not appear to be fitted. Re-fit scaler before using.")

    expected_n = len(scaler.mean_)  # number of features scaler expects

    # If df_features is DataFrame with correct columns:
    if isinstance(df_features, pd.DataFrame):
        X = df_features.copy()
        # If number of columns matches, just use it
        if X.shape[1] == expected_n:
            Xvals = X.values
        else:
            # Try to preserve common columns by name (if names provided)
            # Build an ordered list of columns if available; else create generic feature_i
            if list(X.columns) and all(isinstance(c, str) for c in X.columns):
                # Build aligned columns: existing first, then fill with zeros to match expected_n
                # If number greater than expected, truncate columns (right-most)
                cols = list(X.columns)
                if len(cols) >= expected_n:
                    Xvals = X[cols[:expected_n]].values
                else:
                    # pad
                    pad_count = expected_n - len(cols)
                    pad_array = np.zeros((len(X), pad_count))
                    Xvals = np.hstack([X.values, pad_array])
            else:
                # Columns are not named — create zeros/truncate
                if X.shape[1] >= expected_n:
                    Xvals = X.iloc[:, :expected_n].values
                else:
                    pad_count = expected_n - X.shape[1]
                    pad_array = np.zeros((len(X), pad_count))
                    Xvals = np.hstack([X.values, pad_array])
    else:
        # if input is list-of-lists
        arr = np.array(df_features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == expected_n:
            Xvals = arr
        elif arr.shape[1] > expected_n:
            Xvals = arr[:, :expected_n]
        else:
            pad_count = expected_n - arr.shape[1]
            pad_array = np.zeros((arr.shape[0], pad_count))
            Xvals = np.hstack([arr, pad_array])

    # Now scale with scaler (use transform)
    X_scaled = scaler.transform(Xvals)
    return X_scaled

def predictions_to_labels(preds):
    """
    Map model predictions to readable labels.
    If preds are numeric (0/1) map -> 'normal'/'attack'.
    If preds already strings, return as-is.
    """
    if isinstance(preds, np.ndarray) and preds.dtype.kind in 'iuf':  # numeric
        # assume 0 -> normal, 1 -> attack
        return np.array(["attack" if int(p) == 1 else "normal" for p in preds])
    else:
        # return as strings
        return np.array([str(p) for p in preds])
