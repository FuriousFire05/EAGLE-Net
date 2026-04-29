# src/training/evaluate_model.py
"""Evaluate trained models across clean, corrupted, and latency conditions."""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT

from src.data.dataloader import get_dataloaders
from src.data.eval_conditions import (
    HARD_CLASSES,
    add_gaussian_noise_tensor,
    get_eval_transform,
)
from src.data.eval_transforms import get_unseen_eval_transforms
from src.models.architectures import count_parameters, create_model
from src.utils.config import CONFIG


def get_test_indices():
    """
    Extract the exact same test indices used by the main dataloader.

    Returns:
        Sequence of dataset indices assigned to the test split.
    """
    _, _, test_loader, _ = get_dataloaders()
    return test_loader.dataset.indices


def get_hard_class_indices(class_names):
    """
    Convert hard class names into class indices.

    Args:
        class_names: Ordered class-name list from the EuroSAT dataset.

    Returns:
        Class indices corresponding to configured hard classes.
    """
    return [class_names.index(name) for name in HARD_CLASSES if name in class_names]


def build_condition_loader(condition, test_indices, unseen_transforms):
    """
    Build a test DataLoader for one evaluation condition.

    Args:
        condition: Evaluation condition name.
        test_indices: Dataset indices for the shared test split.
        unseen_transforms: Mapping of extra out-of-distribution transforms.

    Returns:
        Tuple of dataloader, class names, and optional metric label subset.
    """
    data_cfg = CONFIG["data"]

    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    root = data_cfg["root"]

    if condition in unseen_transforms:
        transform = unseen_transforms[condition]
    else:
        transform = get_eval_transform(condition, image_size)

    dataset = EuroSAT(root=root, download=True, transform=transform)
    class_names = dataset.classes

    dataset = Subset(dataset, test_indices)

    metric_labels = None

    if condition == "hard_subset":
        # Restrict evaluation to difficult classes while preserving original
        # class labels for metric computation.
        hard_label_indices = get_hard_class_indices(class_names)
        filtered_indices = []

        for subset_position, original_idx in enumerate(dataset.indices):
            label = dataset.dataset.targets[original_idx]

            if label in hard_label_indices:
                filtered_indices.append(subset_position)

        dataset = Subset(dataset, filtered_indices)
        metric_labels = hard_label_indices

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, class_names, metric_labels


def evaluate_condition(model, loader, condition, device, metric_labels=None):
    """
    Evaluate classification metrics for one condition.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader for the selected evaluation condition.
        condition: Name of the condition being evaluated.
        device: Device used for inference.
        metric_labels: Optional class-label subset for macro metrics.

    Returns:
        Accuracy, precision, recall, F1 score, and confusion matrix.
    """
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            if condition == "noisy":
                # The noisy condition is applied in tensor space to preserve
                # the same base image transform as the clean condition.
                images = add_gaussian_noise_tensor(images)

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
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(CONFIG["model"]["num_classes"])),
    )

    return accuracy, precision, recall, f1, cm


def measure_latency(model, loader, condition, device):
    """
    Measure per-sample inference latency for up to 100 examples.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader used to collect evaluation samples.
        condition: Evaluation condition name.
        device: Device used for inference.

    Returns:
        Mean and median latency in milliseconds.
    """
    samples = []

    # Collect individual samples so each timing measurement reflects a single
    # image inference call rather than batch throughput.
    for images, _ in loader:
        if condition == "noisy":
            images = add_gaussian_noise_tensor(images)

        for i in range(images.size(0)):
            samples.append(images[i].unsqueeze(0))
            if len(samples) >= 100:
                break

        if len(samples) >= 100:
            break

    if not samples:
        return 0.0, 0.0

    # Warm up kernels and caches before measuring latency.
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
    """
    Run multi-condition evaluation for the configured trained model.

    Results are saved as a JSON file in the configured results directory.
    """
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

    image_size = CONFIG["data"]["image_size"]
    test_indices = get_test_indices()
    unseen_transforms = get_unseen_eval_transforms(image_size)

    conditions = [
        "clean",
        "noisy",
        "low_light",
        "blurred",
        "hard_subset",
        "jpeg",
        "color_shift",
        "strong_noise",
        "downscale",
    ]

    all_results = {}

    for condition in conditions:
        print(f"\n=== Evaluating on: {condition} ===")

        # Rebuild the condition-specific test loader so every condition uses
        # the same test split with only the requested transform changed.
        loader, class_names, metric_labels = build_condition_loader(
            condition=condition,
            test_indices=test_indices,
            unseen_transforms=unseen_transforms,
        )

        accuracy, precision, recall, f1, cm = evaluate_condition(
            model=model,
            loader=loader,
            condition=condition,
            device=device,
            metric_labels=metric_labels,
        )

        avg_time, median_time = measure_latency(
            model=model,
            loader=loader,
            condition=condition,
            device=device,
        )

        params = count_parameters(model)
        model_size = os.path.getsize(model_path) / (1024 * 1024)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Latency: {avg_time:.3f} ms")

        result = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "parameters": int(params),
            "model_size_mb": float(model_size),
            "avg_latency_ms": float(avg_time),
            "median_latency_ms": float(median_time),
            "confusion_matrix": cm.tolist(),
        }

        if condition == "hard_subset":
            result["hard_classes"] = HARD_CLASSES
            result["metric_label_indices"] = metric_labels

        all_results[condition] = result

    save_path = os.path.join(results_dir, f"{model_name}_multi_track.json")

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved multi-track results at: {save_path}")


if __name__ == "__main__":
    evaluate()
