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
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, Subset

from src.models.architectures import create_model, count_parameters
from src.utils.config import CONFIG
from src.data.eval_conditions import (
    get_eval_transform,
    add_gaussian_noise_tensor,
    HARD_CLASSES,
)


def get_condition_dataloader(condition):
    data_cfg = CONFIG["data"]

    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    root = data_cfg["root"]

    transform = get_eval_transform(condition, image_size)

    dataset = EuroSAT(root=root, download=True, transform=transform)

    # HARD SUBSET FILTER
    if condition == "hard_subset":
        class_to_idx = dataset.class_to_idx
        selected_indices = []

        for idx, label in enumerate(dataset.targets):
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]
            if class_name in HARD_CLASSES:
                selected_indices.append(idx)

        dataset = Subset(dataset, selected_indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset


def evaluate_condition(model, loader, condition, device):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            if condition == "noisy":
                images = add_gaussian_noise_tensor(images)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, cm, all_labels, all_preds


def measure_latency(model, loader, device):
    samples = []

    for images, _ in loader:
        for i in range(images.size(0)):
            samples.append(images[i].unsqueeze(0))
            if len(samples) >= 100:
                break
        if len(samples) >= 100:
            break

    # warmup
    with torch.no_grad():
        for sample in samples[:10]:
            sample = sample.to(device)
            _ = model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for sample in samples:
            sample = sample.to(device)

            start = time.perf_counter()
            _ = model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

    return float(np.mean(times)), float(np.median(times))


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
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = create_model(model_name, num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"Loaded model from: {model_path}")

    conditions = ["clean", "noisy", "low_light", "blurred", "hard_subset"]

    all_results = {}

    for condition in conditions:
        print(f"\n=== Evaluating on: {condition} ===")

        loader, dataset = get_condition_dataloader(condition)

        acc, precision, recall, f1, cm, y_true, y_pred = evaluate_condition(
            model, loader, condition, device
        )

        avg_time, median_time = measure_latency(model, loader, device)

        params = count_parameters(model)
        model_size = os.path.getsize(model_path) / (1024 * 1024)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Latency: {avg_time:.3f} ms")

        all_results[condition] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "parameters": int(params),
            "model_size_mb": float(model_size),
            "avg_latency_ms": float(avg_time),
            "median_latency_ms": float(median_time),
            "confusion_matrix": cm.tolist(),
        }

    save_path = os.path.join(results_dir, f"{model_name}_multi_track.json")

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved multi-track results at: {save_path}")


if __name__ == "__main__":
    evaluate()