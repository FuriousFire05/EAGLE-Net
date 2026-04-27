"""
Main training script for EAGLE-Net.
Run from terminal: python train.py
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.data.dataset import create_dataloaders, EUROSAT_CLASSES
from src.models.architectures import create_model, count_parameters
from src.training.trainer import Trainer
from src.utils.metrics import MetricsTracker
from src.utils.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_class_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train EAGLE-Net on EuroSAT dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model eager_net --epochs 100 --batch-size 64
  python train.py --model lightweight_cnn --lr 0.0005 --weight-decay 1e-4
  python train.py --model baseline_cnn --epochs 50 --seed 123
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='data/EuroSAT',
                        help='Path to EuroSAT dataset')
    parser.add_argument('--model', type=str, default='eager_net',
                        choices=['baseline_cnn', 'lightweight_cnn', 'eager_net'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu or cuda), auto-detect if not specified')
    parser.add_argument('--save-dir', type=str, default='artifacts',
                        help='Directory to save models and outputs')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed output')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    print("=" * 70)
    print(" EAGLE-Net Training Script")
    print("=" * 70)
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"📍 Device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(f"🔄 Random seed: {args.seed}")
    
    # Check dataset exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n❌ Dataset not found at {data_dir}")
        print(f"Please download EuroSAT and place it in {data_dir}/")
        print(f"Expected structure:")
        print(f"  {data_dir}/")
        print(f"  ├── AnnualCrop/")
        print(f"  ├── Forest/")
        print(f"  └── ... (other class folders)")
        return
    
    # Load dataset
    print("\n📦 Loading dataset...")
    try:
        dataloaders, class_names = create_dataloaders(
            str(data_dir),
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create model
    print(f"\n🏗️  Creating model: {args.model}")
    model = create_model(args.model, num_classes=len(class_names))
    model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"✓ Parameters: {num_params:,}")
    
    # Setup training
    print("\n⚙️  Setting up training...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"✓ Loss: CrossEntropyLoss")
    print(f"✓ Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Create trainer
    save_dir = Path(args.save_dir)
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        class_names=class_names,
    )
    
    # Train
    print(f"\n{'=' * 70}")
    print(f" Training for {args.epochs} epochs")
    print(f"{'=' * 70}\n")
    
    history = trainer.fit(num_epochs=args.epochs, verbose=args.verbose)
    
    # Load best model
    print("\n🔄 Loading best model for evaluation...")
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate on test set
    print(f"\n{'=' * 70}")
    print(" Final Evaluation on Test Set")
    print(f"{'=' * 70}\n")
    
    test_metrics, test_tracker = trainer.evaluate(dataloaders['test'], stage='Test')
    
    # Save results
    print(f"\n{'=' * 70}")
    print(" Saving Results")
    print(f"{'=' * 70}\n")
    
    # Save history
    trainer.save_history('history.json')
    
    # Save test metrics
    test_metrics['class_names'] = class_names
    test_metrics['model'] = args.model
    test_metrics['num_parameters'] = num_params
    
    metrics_path = save_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    
    # Training curves
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        history['train_acc'],
        history['val_acc'],
        save_path=str(save_dir / 'training_curves.png')
    )
    
    # Confusion matrix
    cm = test_tracker.get_confusion_matrix()
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(save_dir / 'confusion_matrix.png'),
        normalize=True
    )
    
    # Class distribution (training set)
    all_targets = []
    for _, labels in dataloaders['train']:
        all_targets.extend(labels.numpy().tolist())
    
    plot_class_distribution(
        all_targets,
        class_names,
        save_path=str(save_dir / 'class_distribution.png')
    )
    
    # Summary
    print(f"\n{'=' * 70}")
    print(" Training Summary")
    print(f"{'=' * 70}\n")
    
    print(f"Model:              {args.model}")
    print(f"Parameters:         {num_params:,}")
    print(f"Epochs:             {args.epochs}")
    print(f"Best Val Accuracy:  {history['best_val_acc']:.4f} (Epoch {history['best_epoch']})")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro):    {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision:     {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall:        {test_metrics['recall_macro']:.4f}")
    print(f"\n✓ All results saved to: {save_dir}/")
    print(f"\n✨ Training complete! To run inference, use app/streamlit_app.py")
    print()


if __name__ == '__main__':
    main()
