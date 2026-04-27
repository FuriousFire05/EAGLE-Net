"""Inference and prediction functions."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

from src.data.dataset import get_data_transforms


class Predictor:
    """
    Inference engine for EuroSAT classification.
    """
    
    def __init__(self, model, class_names, device='cpu', img_size=64):
        """
        Initialize predictor.
        
        Args:
            model (nn.Module): Trained model
            class_names (list): Class names
            device: torch.device
            img_size (int): Input image size
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.img_size = img_size
        
        # Get validation transform (no augmentation)
        _, self.val_transform = get_data_transforms(img_size)
        
        self.model.eval()
    
    def predict_single(self, image_path, return_image=False):
        """
        Predict class for a single image.
        
        Args:
            image_path (str): Path to image file
            return_image (bool): Return preprocessed image tensor
        
        Returns:
            dict: Contains 'class_idx', 'class_name', 'probabilities', 'confidence'
            np.ndarray (optional): Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_original = np.array(image)
        
        # Preprocess
        image_tensor = self.val_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = probs[0, pred_class].item()
        
        result = {
            'class_idx': pred_class.item(),
            'class_name': self.class_names[pred_class.item()],
            'probabilities': probs[0].cpu().numpy(),
            'confidence': confidence,
        }
        
        if return_image:
            return result, image_original
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict for multiple images.
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            list: List of predictions
        """
        results = []
        for img_path in image_paths:
            result = self.predict_single(img_path)
            results.append(result)
        return results
    
    def get_top_k(self, image_path, k=3):
        """
        Get top-k predictions for an image.
        
        Args:
            image_path (str): Path to image
            k (int): Number of top predictions
        
        Returns:
            dict: Contains 'top_classes', 'top_probs'
        """
        result = self.predict_single(image_path)
        probs = result['probabilities']
        
        top_indices = np.argsort(probs)[-k:][::-1]
        top_classes = [self.class_names[i] for i in top_indices]
        top_probs = probs[top_indices]
        
        return {
            'top_classes': top_classes,
            'top_probs': top_probs,
            'top_indices': top_indices,
        }
    
    def predict_from_array(self, image_array):
        """
        Predict from numpy array or PIL Image.
        
        Args:
            image_array: PIL Image or numpy array
        
        Returns:
            dict: Prediction result
        """
        # Convert to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array.astype('uint8')).convert('RGB')
        else:
            image = image_array.convert('RGB')
        
        # Preprocess
        image_tensor = self.val_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = probs[0, pred_class].item()
        
        result = {
            'class_idx': pred_class.item(),
            'class_name': self.class_names[pred_class.item()],
            'probabilities': probs[0].cpu().numpy(),
            'confidence': confidence,
        }
        
        return result


def load_model_for_inference(model_class, checkpoint_path, class_names, device='cpu'):
    """
    Load a trained model for inference.
    
    Args:
        model_class: Model class (e.g., EAGLENet)
        checkpoint_path (str): Path to model checkpoint
        class_names (list): Class names
        device: torch.device
    
    Returns:
        Predictor: Predictor instance
    """
    # Create model
    model = model_class(num_classes=len(class_names))
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Create predictor
    predictor = Predictor(model, class_names, device=device)
    
    return predictor
