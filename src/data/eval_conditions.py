# src/data/eval_conditions.py

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
    """

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
    """
    noise = images.new_empty(images.size()).normal_(0, std)
    noisy = images + noise
    return noisy.clamp(0.0, 1.0)