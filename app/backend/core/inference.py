# app/backend/core/inference.py

import time

import torch
import torch.nn.functional as F

from app.backend.core.model_registry import model_registry

# EuroSAT RGB class order from torchvision.datasets.EuroSAT.
# Keeping this fixed avoids loading/downloading the dataset just to start the API.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def run_inference(model_name: str, input_tensor: torch.Tensor):
    """
    Run inference on a given model and return prediction details.
    """

    model = model_registry.get_model(model_name)
    device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)

    latency = (time.time() - start_time) * 1000  # ms

    confidence, predicted_class = torch.max(probs, dim=1)

    predicted_idx = predicted_class.item()
    confidence_val = confidence.item()

    result = {
        "model": model_name,
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence_val, 4),
        "latency_ms": round(latency, 3),
    }

    return result