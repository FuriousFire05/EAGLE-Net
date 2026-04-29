# app/backend/core/preprocessing.py

import torch
from torchvision import transforms

from src.utils.config import CONFIG


def get_eval_transform():
    """
    Standard evaluation transform (no augmentation).
    """
    image_size = CONFIG["data"]["image_size"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def preprocess_image(image):
    """
    Convert PIL image to tensor ready for model.
    """
    transform = get_eval_transform()
    tensor = transform(image).unsqueeze(0)  # add batch dim
    return tensor