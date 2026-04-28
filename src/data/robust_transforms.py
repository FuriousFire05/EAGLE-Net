# src/data/robust_transforms.py

import random

import torch
from torchvision import transforms


class AddGaussianNoise:
    """
    Mild Gaussian noise applied after ToTensor.
    Keeps pixel values clamped to [0, 1].
    """

    def __init__(self, std=0.04):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomGamma:
    """
    Mild gamma/brightness style perturbation after ToTensor.
    Useful for low-light robustness.
    """

    def __init__(self, gamma_range=(0.8, 1.25)):
        self.gamma_range = gamma_range

    def __call__(self, tensor):
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