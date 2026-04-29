# src/data/eval_conditions.py
"""Evaluation conditions used for multi-track robustness reporting."""

from torchvision import transforms


HARD_CLASSES = [
    "HerbaceousVegetation",
    "PermanentCrop",
    "River",
    "Highway",
]


def get_eval_transform(condition: str, image_size: int):
    """
    Returns evaluation transform for different simulated deployment conditions.

    Args:
        condition: Name of the evaluation condition to simulate.
        image_size: Target square image size for model input.

    Returns:
        Composed torchvision transform for the requested condition.

    Raises:
        ValueError: If the condition name is not recognized.
    """

    # All conditions begin with the same resize step so downstream metrics are
    # comparable across clean and corrupted inputs.
    base = [
        transforms.Resize((image_size, image_size)),
    ]

    if condition == "clean":
        extra = []

    elif condition == "noisy":
        # noise will be applied later in tensor space (NOT here)
        extra = []

    elif condition == "low_light":
        extra = [
            transforms.ColorJitter(brightness=(0.7, 0.9)),
        ]

    elif condition == "blurred":
        extra = [
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)),
        ]

    elif condition == "hard_subset":
        extra = []

    else:
        raise ValueError(
            f"Unknown condition: {condition}. "
            "Choose from: clean, noisy, low_light, blurred, hard_subset"
        )

    return transforms.Compose(base + extra + [transforms.ToTensor()])


def add_gaussian_noise_tensor(images, std=0.05):
    """
    Adds Gaussian noise after tensor conversion.

    Args:
        images: Batch of image tensors in the range [0, 1].
        std: Standard deviation of the Gaussian noise.

    Returns:
        Noisy image batch clamped to [0, 1].
    """
    noise = images.new_empty(images.size()).normal_(0, std)
    noisy = images + noise
    return noisy.clamp(0.0, 1.0)
