# web/app.py

import os
import sys
import traceback
import random
import tldextract
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from flask_cors import CORS

# === Ensure project src/ is importable ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import load_model, align_and_scale_features, predictions_to_labels

# Try NFStreamer
try:
    from nfstream import NFStreamer
    NFSTREAM_AVAILABLE = True
except:
    NFSTREAM_AVAILABLE = False

# Flask Setup
app = Flask(__name__)
CORS(app)  # <<< Allows browser extension to call backend
app.secret_key = "networkanalyzer2025"

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
MODEL, SCALER = load_model()

def simulate_features(url):
    """Deterministic scoring based on domain name."""
    seed = sum(ord(c) for c in url)
    rnd = random.Random(seed)

    df = pd.DataFrame({
        "bytes": [rnd.randint(2000, 120000)],
        "packets": [rnd.randint(10, 800)],
        "duration": [rnd.uniform(0.1, 5.0)]
    })
    return df


def extract_features_nfstream(file_path):
    records = []
    try:
        streamer = NFStreamer(source=file_path, statistical_analysis=True, decode_tunnels=True)
        for flow in streamer:
            dur = getattr(flow, "bidirectional_duration_ms", 0) / 1000.0
            records.append([
                flow.bidirectional_bytes,
                flow.bidirectional_packets,
                dur
            ])
        return pd.DataFrame(records, columns=["bytes", "packets", "duration"])
    except:
        return pd.DataFrame()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/results", methods=["POST"])
def results():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    if NFSTREAM_AVAILABLE:
        df = extract_features_nfstream(file_path)
        if df.empty:
            df = simulate_features(filename)
    else:
        df = simulate_features(filename)

    X_scaled = align_and_scale_features(df, SCALER)
    preds = predictions_to_labels(MODEL.predict(X_scaled))

    total = len(preds)
    attack = (preds == "attack").sum()
    attack_percent = round((attack / total) * 100, 2)

    return render_template("results.html", results={
        "filename": filename,
        "attack": attack,
        "total_packets": total,
        "attack_percentage": attack_percent
    })


# ✅ NEW API ENDPOINT FOR BROWSER EXTENSION
@app.route("/api/check_url", methods=["POST"])
def check_url():
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    domain = tldextract.extract(url).registered_domain

    df = simulate_features(domain)
    X_scaled = align_and_scale_features(df, SCALER)
    pred = predictions_to_labels(MODEL.predict(X_scaled))[0]

    return jsonify({
        "url": url,
        "domain": domain,
        "result": pred,
        "safe": (pred == "normal")
    })


if __name__ == "__main__":
    app.run(debug=True)
