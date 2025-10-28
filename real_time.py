import argparse
import pandas as pd
from nfstream import NFStreamer
from utils import load_model
from sklearn.preprocessing import StandardScaler

def extract_features_from_flow(flow):
    return [flow.bidirectional_bytes, flow.bidirectional_packets, flow.duration]

def main(interface):
    model, scaler = load_model()
    print(f"✅ Capturing live traffic from: {interface}")

    try:
        streamer = NFStreamer(source=interface, statistical_analysis=True, decode_tunnels=True)
    except Exception as e:
        print(f"❌ Could not start NFStreamer: {e}")
        return

    for flow in streamer:
        try:
            features = extract_features_from_flow(flow)
            df = pd.DataFrame([features], columns=["bytes", "packets", "duration"])
            local_scaler = StandardScaler()
            df_scaled = local_scaler.fit_transform(df)
            try:
                pred = model.predict(df_scaled)
            except Exception:
                pred = ["unknown"]
            print(f"Flow {flow.src_ip} → {flow.dst_ip} | Prediction: {pred[0]}")
        except Exception as e:
            print("Flow error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True, help="e.g. Wi-Fi or Ethernet")
    args = parser.parse_args()
    main(args.interface)

