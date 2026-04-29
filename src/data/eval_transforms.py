# src/data/eval_transforms.py
"""Evaluation-only image corruptions for robustness testing."""

import io
import random

import torch
from PIL import Image
from torchvision import transforms


class JPEGCompression:
    """
    Simulates unseen JPEG compression artifacts.
    This is not used during training.

    Args:
        quality_range: Inclusive JPEG quality range sampled per image.
    """

    def __init__(self, quality_range=(10, 40)):
        """Store the quality range used when compressing images."""
        self.quality_range = quality_range

    def __call__(self, img):
        """Apply JPEG compression and return a decoded RGB image."""
        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class StrongGaussianNoise:
    """
    Stronger unseen noise than the mild training noise.
    Applied after ToTensor.

    Args:
        std: Standard deviation of the Gaussian noise.
    """

    def __init__(self, std=0.10):
        """Store the noise scale applied to tensors."""
        self.std = std

    def __call__(self, tensor):
        """Add Gaussian noise and clamp tensor values to the image range."""
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_unseen_eval_transforms(image_size):
    """
    Evaluation-only distribution-shifted corruptions.

    These are intentionally different from the robustness training transforms
    so we can test out-of-distribution robustness.

    Args:
        image_size: Target square image size for evaluation.

    Returns:
        Dictionary mapping condition names to torchvision transforms.
    """

    return {
        "jpeg": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            JPEGCompression(quality_range=(10, 40)),
            transforms.ToTensor(),
        ]),

        "color_shift": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=0.30,
                contrast=0.30,
                saturation=0.30,
                hue=0.20,
            ),
            transforms.ToTensor(),
        ]),

        "strong_noise": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            StrongGaussianNoise(std=0.10),
        ]),

        "downscale": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]),
    }
