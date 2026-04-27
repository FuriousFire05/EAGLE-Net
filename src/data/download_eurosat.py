"""
Utility script to download and prepare EuroSAT dataset.
Run: python src/data/download_eurosat.py
"""

import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
import argparse


def download_eurosat(save_dir='data/EuroSAT'):
    """
    Download EuroSAT dataset using torchvision.
    
    Args:
        save_dir (str): Directory to save dataset
    """
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading EuroSAT dataset to {save_dir}...")
    print("(This may take a few minutes and ~600 MB)")
    
    try:
        # Download
        dataset = EuroSAT(
            root=str(save_dir),
            download=True,
            split='train',  # torchvision provides train/val splits
            transform=None
        )
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Classes: {len(dataset.classes)}")
        
        # Print class info
        print(f"\nClass mapping:")
        for idx, class_name in enumerate(dataset.classes):
            print(f"  {idx}: {class_name}")
        
        print(f"\n✓ Dataset structure created at: {save_dir}")
        print(f"  Classes are organized in subdirectories.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"\nAlternative: Download manually from:")
        print(f"  https://madm.web.unc.edu/datasets/remote_sensing_datasets/#eurosat")
        return False


def verify_dataset(save_dir='data/EuroSAT'):
    """
    Verify dataset integrity.
    
    Args:
        save_dir (str): Path to dataset
    """
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        print(f"❌ Dataset not found at {save_dir}")
        return False
    
    # Count files
    print(f"📊 Dataset Verification:")
    print(f"  Location: {save_dir}")
    
    total_images = 0
    class_counts = {}
    
    for class_dir in save_dir.iterdir():
        if class_dir.is_dir():
            image_count = len(list(class_dir.glob('*.jpg')))
            if image_count == 0:
                image_count = len(list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = image_count
            total_images += image_count
    
    if not class_counts:
        print(f"❌ No images found in {save_dir}")
        return False
    
    print(f"\n✓ Dataset found:")
    print(f"  Total images: {total_images}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"\n  Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"    {class_name:.<20} {count:>5} images")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download/verify EuroSAT dataset')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--verify', action='store_true', help='Verify existing dataset')
    parser.add_argument('--data-dir', type=str, default='data/EuroSAT',
                        help='Dataset directory path')
    
    args = parser.parse_args()
    
    if args.download:
        download_eurosat(args.data_dir)
    
    if args.verify or not args.download:
        verify_dataset(args.data_dir)


if __name__ == '__main__':
    main()
