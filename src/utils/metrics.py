"""Metrics computation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


class MetricsTracker:
    """Track and compute metrics during training/validation."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        """
        Update with batch predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions, shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels, shape (batch_size,)
        """
        # Convert to numpy
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Get class predictions
        pred_classes = np.argmax(preds_np, axis=1)
        
        self.predictions.extend(pred_classes.tolist())
        self.targets.extend(targets_np.tolist())
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Metrics including accuracy, F1, precision, recall
        """
        if len(self.predictions) == 0:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(self.targets, self.predictions),
            'f1_macro': f1_score(self.targets, self.predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(self.targets, self.predictions, average='weighted', zero_division=0),
            'precision_macro': precision_score(self.targets, self.predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(self.targets, self.predictions, average='weighted', zero_division=0),
            'recall_macro': recall_score(self.targets, self.predictions, average='macro', zero_division=0),
            'recall_weighted': recall_score(self.targets, self.predictions, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix."""
        return confusion_matrix(self.targets, self.predictions, labels=range(self.num_classes))
    
    def get_classification_report(self, class_names=None):
        """Get detailed classification report."""
        return classification_report(
            self.targets,
            self.predictions,
            target_names=class_names,
            digits=4,
            zero_division=0
        )


def print_metrics(metrics, stage=''):
    """Pretty print metrics."""
    print(f"\n{stage} Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 (macro):      {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 (weighted):   {metrics.get('f1_weighted', 0):.4f}")
    print(f"  Precision (macro):    {metrics.get('precision_macro', 0):.4f}")
    print(f"  Precision (weighted): {metrics.get('precision_weighted', 0):.4f}")
    print(f"  Recall (macro):       {metrics.get('recall_macro', 0):.4f}")
    print(f"  Recall (weighted):    {metrics.get('recall_weighted', 0):.4f}")
