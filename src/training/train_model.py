import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.architectures import create_model
from src.data.dataloader import get_dataloaders
from src.utils.config import CONFIG

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Config
model_name = CONFIG["model"]["name"]
batch_size = CONFIG["training"]["batch_size"]
epochs = CONFIG["training"]["epochs"]
lr = CONFIG["training"]["learning_rate"]

# Data
train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

# Model
model = create_model(model_name).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# Scheduler (important)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Save setup
os.makedirs("artifacts/models", exist_ok=True)
best_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # ---- TRAIN ----
    model.train()
    train_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        save_path = f"artifacts/models/{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved BEST model at {save_path}")

    # Step scheduler
    scheduler.step()

print("\nTraining complete!")
print(f"Best Validation Accuracy: {best_acc:.2f}%")