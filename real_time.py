import argparse
from nfstream import NFStreamer
from utils import load_model
import pandas as pd

def extract_features_from_flow(flow):
    """Convert NFStreamer flow object to ML feature vector"""
    # Example: basic features, extend as needed
    return [
        flow.bidirectional_bytes,
        flow.bidirectional_packets,
        flow.duration
    ]

def main(interface):
    model, scaler = load_model()
    print(f"Starting live capture on {interface}...")
    streamer = NFStreamer(source=interface, statistical_analysis=True, decode_tunnels=True)
    
    for flow in streamer:
        features = extract_features_from_flow(flow)
        df = pd.DataFrame([features])
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)
        print(f"Flow: {flow.src_ip} -> {flow.dst_ip} | Prediction: {pred[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True, help="Network interface to capture (e.g., Wi-Fi)")
    args = parser.parse_args()
    main(args.interface)

