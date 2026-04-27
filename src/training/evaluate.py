"""
Evaluation script for trained models.
Run: python evaluate.py --model-path artifacts/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path

from src.data.dataset import create_dataloaders, EUROSAT_CLASSES
from src.models.architectures import create_model, count_parameters
from src.training.trainer import Trainer
from src.utils.metrics import MetricsTracker
from src.utils.visualization import (
    plot_confusion_matrix, plot_class_distribution
)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained EAGLE-Net model')
    parser.add_argument('--model', type=str, default='eager_net',
                        choices=['baseline_cnn', 'lightweight_cnn', 'eager_net'],
                        help='Model architecture')
    parser.add_argument('--model-path', type=str, default='artifacts/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/EuroSAT',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu or cuda)')
    parser.add_argument('--save-dir', type=str, default='artifacts',
                        help='Directory to save evaluation results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 70)
    print(" EAGLE-Net Evaluation Script")
    print("=" * 70)
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"📍 Device: {device}")
    
    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Check dataset exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Dataset not found at {data_dir}")
        return
    
    # Load dataset
    print("\n📦 Loading dataset...")
    class_names = list(EUROSAT_CLASSES.values())
    
    try:
        dataloaders, _ = create_dataloaders(
            str(data_dir),
            batch_size=args.batch_size,
            seed=42,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create model
    print(f"\n🏗️  Creating model: {args.model}")
    model = create_model(args.model, num_classes=len(class_names))
    model.to(device)
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"✓ Parameters: {num_params:,}")
    
    # Setup trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        class_names=class_names,
    )
    
    # Evaluate
    print(f"\n{'=' * 70}")
    print(f" Evaluating on {args.split.upper()} Set")
    print(f"{'=' * 70}\n")
    
    split_loader = dataloaders[args.split]
    metrics, tracker = trainer.evaluate(split_loader, stage=args.split.upper())
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(" Saving Evaluation Results")
    print(f"{'=' * 70}\n")
    
    # Save metrics
    metrics['class_names'] = class_names
    metrics['model'] = args.model
    metrics['num_parameters'] = num_params
    metrics['split'] = args.split
    
    metrics_path = save_dir / f'evaluation_{args.split}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Save confusion matrix
    cm = tracker.get_confusion_matrix()
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(save_dir / f'confusion_matrix_{args.split}.png'),
        normalize=True
    )
    
    # Save classification report
    report_path = save_dir / f'classification_report_{args.split}.txt'
    report = tracker.get_classification_report(class_names)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved classification report: {report_path}")
    
    print(f"\n✓ Evaluation complete!")
    print()


if __name__ == '__main__':
    main()
