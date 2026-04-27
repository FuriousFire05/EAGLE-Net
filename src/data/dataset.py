"""EuroSAT RGB dataset loading and preprocessing."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path


# EuroSAT class mapping
EUROSAT_CLASSES = {
    0: 'Annual Crop',
    1: 'Forest',
    2: 'Herbaceous Vegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'Permanent Crop',
    7: 'Residential',
    8: 'River',
    9: 'Sea Lake'
}

NUM_CLASSES = len(EUROSAT_CLASSES)


class EuroSATDataset(Dataset):
    """
    Custom EuroSAT RGB dataset loader.
    
    Expects data organized in folders:
    data/EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── ...
    └── SeaLake/
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize dataset.
        
        Args:
            root_dir (str): Path to dataset root containing class folders
            transform (callable): Image transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Class name to index mapping
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append(str(img_path))
                    self.targets.append(class_idx)
                for img_path in class_dir.glob('*.png'):
                    self.samples.append(str(img_path))
                    self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get image and label."""
        img_path = self.samples[idx]
        label = self.targets[idx]
        
        # Load RGB image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(img_size=64):
    """
    Return train and validation transforms.
    
    Args:
        img_size (int): Image size (default 64 for EuroSAT)
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Normalization values (ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def create_dataloaders(
    data_dir,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    num_workers=0,
    img_size=64,
):
    """
    Create train/val/test DataLoaders with reproducible split.
    
    Args:
        data_dir (str): Path to dataset root
        batch_size (int): Batch size
        train_ratio (float): Proportion for training (default 0.7)
        val_ratio (float): Proportion for validation (default 0.15)
        test_ratio (float): Proportion for testing (default 0.15)
        seed (int): Random seed for reproducibility
        num_workers (int): Number of data loading workers
        img_size (int): Image size (default 64)
    
    Returns:
        dict: Contains 'train', 'val', 'test' DataLoaders
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(img_size)
    
    # Load full dataset
    full_dataset = EuroSATDataset(data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms
    train_dataset.dataset = EuroSATDataset(data_dir, transform=train_transform)
    train_dataset.indices = train_dataset.indices
    
    val_dataset.dataset = EuroSATDataset(data_dir, transform=val_transform)
    val_dataset.indices = val_dataset.indices
    
    test_dataset.dataset = EuroSATDataset(data_dir, transform=val_transform)
    test_dataset.indices = test_dataset.indices
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
    }
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")
    
    return dataloaders, full_dataset.class_names
