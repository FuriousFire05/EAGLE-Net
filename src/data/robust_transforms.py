# src/data/robust_transforms.py
"""Training-time robustness augmentations for satellite image classification."""

import random

import torch
from torchvision import transforms


class AddGaussianNoise:
    """
    Mild Gaussian noise applied after ToTensor.
    Keeps pixel values clamped to [0, 1].

    Args:
        std: Standard deviation of the Gaussian noise.
    """

    def __init__(self, std=0.04):
        """Store the noise scale applied to tensors."""
        self.std = std

    def __call__(self, tensor):
        """Add noise to a tensor and clamp values to the valid image range."""
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomGamma:
    """
    Mild gamma/brightness style perturbation after ToTensor.
    Useful for low-light robustness.

    Args:
        gamma_range: Range sampled to exponentiate image tensors.
    """

    def __init__(self, gamma_range=(0.8, 1.25)):
        """Store the random gamma sampling range."""
        self.gamma_range = gamma_range

    def __call__(self, tensor):
        """Apply a sampled gamma correction and clamp the output tensor."""
        gamma = random.uniform(*self.gamma_range)
        return torch.clamp(tensor ** gamma, 0.0, 1.0)


def get_robust_train_transform(image_size):
    """
    Robust training transform for EAGLE-Net v3.

    Designed to simulate mild deployment degradations:
    - scale/crop variation
    - flips and rotations
    - brightness/contrast shifts
    - mild blur
    - mild noise
    - random erasing

    Args:
        image_size: Target square crop size for model input.

    Returns:
        Composed torchvision transform for robust training.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),

        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.08,
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 0.8)),
        ], p=0.25),

        transforms.ToTensor(),

        transforms.RandomApply([
            AddGaussianNoise(std=0.04),
        ], p=0.25),

        transforms.RandomApply([
            RandomGamma(gamma_range=(0.8, 1.25)),
        ], p=0.20),

        transforms.RandomErasing(
            p=0.15,
            scale=(0.01, 0.05),
            ratio=(0.5, 2.0),
            value="random",
        ),
    ])
