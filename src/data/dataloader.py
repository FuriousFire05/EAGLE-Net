# src/data/dataloader.py
"""Dataset transforms and DataLoader construction for EuroSAT experiments."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

from src.data.robust_transforms import get_robust_train_transform
from src.utils.config import CONFIG


def get_transforms():
    """
    Build train/evaluation transforms.

    Standard training is used for baseline/lightweight models.
    Robust training is used when enabled in config, mainly for EAGLE-Net.

    Returns:
        Tuple containing the training transform and evaluation transform.
    """

    image_size = CONFIG["data"]["image_size"]
    model_name = CONFIG["model"]["name"]
    robust_training = CONFIG["training"].get("robust_training", False)

    standard_train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])

    robust_train_transform = get_robust_train_transform(image_size)

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Keep the training augmentation policy controlled by CONFIG so the same
    # dataloader function can support baseline and robust experiments.
    if robust_training:
        train_transform = robust_train_transform
    else:
        train_transform = standard_train_transform

    return train_transform, eval_transform


def get_dataloaders():
    """
    Create reproducible EuroSAT train/validation/test dataloaders.

    Returns:
        Tuple of train loader, validation loader, test loader, and class names.
    """

    data_cfg = CONFIG["data"]

    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    seed = data_cfg["seed"]
    root = data_cfg["root"]

    train_transform, eval_transform = get_transforms()

    full_dataset_for_split = EuroSAT(
        root=root,
        download=True,
        transform=None,
    )

    total_size = len(full_dataset_for_split)

    # Split once with a seeded generator so each model evaluates on the same
    # train/validation/test partitions.
    train_size = int(data_cfg["train_split"] * total_size)
    val_size = int(data_cfg["val_split"] * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset, test_subset = random_split(
        full_dataset_for_split,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # Rebuild datasets with split-specific transforms, then reuse the exact
    # indices generated above to avoid transform leakage across splits.
    train_dataset = EuroSAT(
        root=root,
        download=False,
        transform=train_transform,
    )

    val_dataset = EuroSAT(
        root=root,
        download=False,
        transform=eval_transform,
    )

    test_dataset = EuroSAT(
        root=root,
        download=False,
        transform=eval_transform,
    )

    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_names = full_dataset_for_split.classes

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
