"""Visualization utilities for plots and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    return fig


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Class name labels
        save_path (str): Path to save figure
        normalize (bool): Normalize by row (True) or show counts (False)
    """
    if normalize:
        # Normalize by row (precision per class)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        data = cm_norm
        fmt = '.2%'
    else:
        data = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    return fig


def plot_class_distribution(targets, class_names, save_path=None):
    """
    Plot class distribution.
    
    Args:
        targets (list or np.ndarray): Target labels
        class_names (list): Class name labels
        save_path (str): Path to save figure
    """
    unique, counts = np.unique(targets, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_pos = np.arange(len(class_names))
    ax.bar(x_pos, [counts[i] if i in unique else 0 for i in range(len(class_names))],
           color='steelblue', edgecolor='navy', alpha=0.7)
    
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate([counts[i] if i in unique else 0 for i in range(len(class_names))]):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved class distribution to {save_path}")
    
    return fig


def plot_top_predictions(image, probabilities, class_names, top_k=3, save_path=None):
    """
    Plot image with top-k predicted classes.
    
    Args:
        image (np.ndarray): Image array (H x W x C)
        probabilities (np.ndarray): Class probabilities
        class_names (list): Class names
        top_k (int): Top-k classes to show
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    
    # Top-k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    colors = ['gold', 'silver', '#CD7F32']  # Gold, Silver, Bronze
    axes[1].barh(range(top_k), top_probs, color=colors[:top_k])
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(top_classes)
    axes[1].set_xlabel('Probability', fontsize=11, fontweight='bold')
    axes[1].set_title('Top-3 Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1])
    
    # Add probability values
    for i, prob in enumerate(top_probs):
        axes[1].text(prob + 0.02, i, f'{prob:.2%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved prediction visualization to {save_path}")
    
    return fig
