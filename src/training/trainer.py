"""Training loop and trainer class."""

import torch
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm

from src.utils.metrics import MetricsTracker, print_metrics


class Trainer:
    """
    Training manager for EuroSAT classification.
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Metrics tracking
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        save_dir='artifacts',
        class_names=None,
    ):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): PyTorch model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: torch.device (cpu or cuda)
            save_dir (str): Directory to save checkpoints
            class_names (list): Class name labels
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0,
            'best_epoch': 0,
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        metrics = MetricsTracker(num_classes=len(self.class_names) if self.class_names else 10)
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs, labels)
            
            pbar.set_postfix({'loss': loss.item():.4f})
        
        avg_loss = total_loss / len(self.train_loader)
        metrics_dict = metrics.compute()
        
        return avg_loss, metrics_dict.get('accuracy', 0)
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        metrics = MetricsTracker(num_classes=len(self.class_names) if self.class_names else 10)
        
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                metrics.update(outputs, labels)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics_dict = metrics.compute()
        
        return avg_loss, metrics_dict.get('accuracy', 0), metrics
    
    def fit(self, num_epochs, verbose=True):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs
            verbose (bool): Print progress
        
        Returns:
            dict: Training history
        """
        print(f"\n📊 Starting training for {num_epochs} epochs...")
        print(f"📍 Device: {self.device}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.history['best_val_acc']:
                self.history['best_val_acc'] = val_acc
                self.history['best_epoch'] = epoch + 1
                self.save_checkpoint('best_model.pt')
            
            # Print
            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        print(f"\n✓ Training complete!")
        print(f"  Best validation accuracy: {self.history['best_val_acc']:.4f} (Epoch {self.history['best_epoch']})")
        
        return self.history
    
    def evaluate(self, data_loader, stage='Test'):
        """
        Evaluate on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            stage (str): Stage name (e.g., 'Test')
        
        Returns:
            dict: Metrics
        """
        self.model.eval()
        metrics = MetricsTracker(num_classes=len(self.class_names) if self.class_names else 10)
        
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'{stage} Evaluation', leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                metrics.update(outputs, labels)
        
        avg_loss = total_loss / len(data_loader)
        metrics_dict = metrics.compute()
        metrics_dict['loss'] = avg_loss
        
        print_metrics(metrics_dict, stage=f'{stage} Set')
        
        # Print classification report
        if self.class_names:
            print(f"\n{stage} Classification Report:")
            print(metrics.get_classification_report(self.class_names))
        
        return metrics_dict, metrics
    
    def save_checkpoint(self, filename='model.pt'):
        """Save model checkpoint."""
        save_path = self.save_dir / filename
        torch.save(self.model.state_dict(), save_path)
        if filename == 'best_model.pt':
            print(f"✓ Saved best model: {save_path}")
    
    def load_checkpoint(self, filename='best_model.pt'):
        """Load model checkpoint."""
        load_path = self.save_dir / filename
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"✓ Loaded model: {load_path}")
    
    def save_history(self, filename='history.json'):
        """Save training history."""
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved history: {save_path}")
