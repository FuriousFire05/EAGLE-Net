import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os

from src.models.architectures import create_model, count_parameters
from src.data.dataloader import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_name="lightweight_cnn", model_path=None):

    # Load data
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Load model
    model = create_model(model_name).to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from:", model_path)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    cm = confusion_matrix(all_labels, all_preds)

    # Params
    params = count_parameters(model)

    # Model size
    if model_path and os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
    else:
        size_mb = None

    print("\n=== Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Parameters: {params}")

    if size_mb:
        print(f"Model Size: {size_mb:.2f} MB")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "params": params,
        "model_size_mb": size_mb
    }


if __name__ == "__main__":
    evaluate("baseline_cnn", "artifacts/models/baseline_cnn.pth")