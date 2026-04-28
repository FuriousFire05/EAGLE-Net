# src/training/evaluate_model.py

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataloader import get_dataloaders
from src.models.architectures import count_parameters, create_model
from src.utils.config import CONFIG


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_name = CONFIG["model"]["name"]
    num_classes = CONFIG["model"]["num_classes"]

    model_dir = CONFIG["paths"]["model_dir"]
    results_dir = CONFIG["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Train the model first using: python -m src.training.train_model"
        )

    _, _, test_loader, class_names = get_dataloaders()

    model = create_model(model_name, num_classes=num_classes).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from: {model_path}")

    all_preds = []
    all_labels = []

    # ========================
    # STANDARD EVALUATION
    # ========================
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    # ========================
    # EFFICIENCY METRICS
    # ========================
    params = count_parameters(model)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Inference timing: batch-1 latency
    timing_images = []
    with torch.no_grad():
        for images, _ in test_loader:
            for i in range(images.size(0)):
                timing_images.append(images[i].unsqueeze(0))
                if len(timing_images) >= 100:
                    break
            if len(timing_images) >= 100:
                break

    # Warmup
    with torch.no_grad():
        for sample in timing_images[:10]:
            sample = sample.to(device)
            _ = model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []

    with torch.no_grad():
        for sample in timing_images:
            sample = sample.to(device)

            start = time.perf_counter()
            _ = model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    avg_inference_time_ms = float(np.mean(latencies_ms))
    median_inference_time_ms = float(np.median(latencies_ms))

    # ========================
    # PRINT RESULTS
    # ========================
    print("\n=== Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Parameters: {params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Avg Inference Time: {avg_inference_time_ms:.3f} ms/image")
    print(f"Median Inference Time: {median_inference_time_ms:.3f} ms/image")

    print("\n=== Classification Report ===")
    print(report_text)

    print("\n=== Confusion Matrix ===")
    print(cm)

    # ========================
    # SAVE RESULTS
    # ========================
    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "parameters": int(params),
        "model_size_mb": float(model_size_mb),
        "avg_inference_time_ms": avg_inference_time_ms,
        "median_inference_time_ms": median_inference_time_ms,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    results_path = os.path.join(results_dir, f"{model_name}_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved at: {results_path}")


if __name__ == "__main__":
    evaluate()