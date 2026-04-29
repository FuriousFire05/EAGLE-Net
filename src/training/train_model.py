# src/training/train_model.py
"""Train the configured EuroSAT classification model."""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from src.models.architectures import create_model
from src.data.dataloader import get_dataloaders
from src.utils.config import CONFIG


def train():
    """
    Train the configured model and save the best checkpoint plus history.

    The active architecture, optimizer settings, scheduler settings, and paths
    are read from ``CONFIG``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========================
    # CONFIG
    # ========================
    model_name = CONFIG["model"]["name"]
    num_classes = CONFIG["model"]["num_classes"]

    batch_size = CONFIG["data"]["batch_size"]

    epochs = CONFIG["training"]["epochs"]
    lr = CONFIG["training"]["learning_rate"]
    weight_decay = CONFIG["training"]["weight_decay"]
    step_size = CONFIG["training"]["scheduler_step_size"]
    gamma = CONFIG["training"]["scheduler_gamma"]

    model_dir = CONFIG["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # ========================
    # DATA
    # ========================
    train_loader, val_loader, _, _ = get_dataloaders()

    # ========================
    # MODEL
    # ========================
    model = create_model(model_name, num_classes=num_classes).to(device)

    # ========================
    # LOSS + OPTIMIZER
    # ========================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # ========================
    # TRAINING LOOP
    # ========================
    best_acc = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Standard supervised classification step: forward pass, loss,
            # gradient reset, backpropagation, and optimizer update.
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Validation runs without gradients and tracks aggregate
                # accuracy across the full validation split.
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # ---- LOG ----
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.2f}%")

        # ---- SAVE BEST MODEL ----
        # Only the strongest validation checkpoint is kept for later evaluation.
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_path = os.path.join(model_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved BEST model at {save_path}")

        scheduler.step()

    # ========================
    # SAVE TRAINING HISTORY
    # ========================
    history_path = os.path.join(model_dir, f"{model_name}_history.json")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"History saved at {history_path}")


if __name__ == "__main__":
    train()
