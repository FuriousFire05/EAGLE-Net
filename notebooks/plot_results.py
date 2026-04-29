import json
import os
import matplotlib.pyplot as plt

# -------- Load data --------

def load_results(name):
    path = f"artifacts/results/{name}_multi_track.json"
    with open(path, "r") as f:
        return json.load(f)

baseline = load_results("baseline_cnn")
lightweight = load_results("lightweight_cnn")
eagle = load_results("eagle_net")

# -------- Conditions --------

conditions = list(eagle.keys())

# -------- Extract metrics --------

def get_metric(data, metric):
    return [data[c][metric] for c in conditions]

# -------- Plot 1: Accuracy --------

plt.figure(figsize=(10,5))

plt.plot(conditions, get_metric(baseline, "accuracy"), marker='o', label="BaselineCNN")
plt.plot(conditions, get_metric(lightweight, "accuracy"), marker='o', label="LightweightCNN")
plt.plot(conditions, get_metric(eagle, "accuracy"), marker='o', label="EAGLE-Net")

plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Across Conditions")
plt.legend()
plt.grid(True)

os.makedirs("artifacts/plots", exist_ok=True)
plt.tight_layout()
plt.savefig("artifacts/plots/accuracy_comparison.png")

print("Saved: accuracy_comparison.png")

# -------- Plot 2: F1 --------

plt.figure(figsize=(10,5))

plt.plot(conditions, get_metric(baseline, "f1"), marker='o', label="BaselineCNN")
plt.plot(conditions, get_metric(lightweight, "f1"), marker='o', label="LightweightCNN")
plt.plot(conditions, get_metric(eagle, "f1"), marker='o', label="EAGLE-Net")

plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.title("Model F1 Score Across Conditions")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("artifacts/plots/f1_comparison.png")

print("Saved: f1_comparison.png")

# -------- Plot 3: Latency vs Accuracy --------

plt.figure()

def get_clean_acc(data):
    return data["clean"]["accuracy"]

def get_latency(data):
    return data["clean"]["avg_latency_ms"]

plt.scatter(get_latency(baseline), get_clean_acc(baseline), label="BaselineCNN")
plt.scatter(get_latency(lightweight), get_clean_acc(lightweight), label="LightweightCNN")
plt.scatter(get_latency(eagle), get_clean_acc(eagle), label="EAGLE-Net")

plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy")
plt.title("Latency vs Accuracy Tradeoff")
plt.legend()
plt.grid(True)

plt.savefig("artifacts/plots/latency_tradeoff.png")

print("Saved: latency_tradeoff.png")