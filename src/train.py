import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = r"C:\Users\AVADUT\RealTime_Network_Analyzer"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === If you have a sample.pcap in data/raw you could extract real flows here.
# For now we create a synthetic dataset with labels (0 normal, 1 attack)
n = 1000
bytes_ = np.random.randint(200, 200000, n)
packets = np.random.randint(1, 2000, n)
duration = np.random.uniform(0.001, 20.0, n)

# Create labels: make longer flows more likely "attack" for demo purposes
labels = ( (bytes_ > 100000) | (packets > 1500) ).astype(int)

X = pd.DataFrame({"bytes": bytes_, "packets": packets, "duration": duration})
y = labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Saved model and scaler to:", MODEL_DIR)
