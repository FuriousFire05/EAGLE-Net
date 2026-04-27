"""
Inference script for predictions on images.
Run: python inference.py --image-path path/to/image.jpg
"""

import argparse
import torch
from pathlib import Path
import numpy as np

from src.models.architectures import create_model
from src.inference.predictor import Predictor
from src.data.dataset import EUROSAT_CLASSES
from src.utils.visualization import plot_top_predictions


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on EuroSAT images')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='eager_net',
                        choices=['baseline_cnn', 'lightweight_cnn', 'eager_net'],
                        help='Model architecture')
    parser.add_argument('--model-path', type=str, default='artifacts/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu or cuda)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to show')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save prediction plot')
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 70)
    print(" EAGLE-Net Inference Script")
    print("=" * 70)
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"📍 Device: {device}")
    
    # Check files
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Image not found at {image_path}")
        return
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load model
    print(f"\n🏗️  Loading model: {args.model}")
    class_names = list(EUROSAT_CLASSES.values())
    
    model = create_model(args.model, num_classes=len(class_names))
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Create predictor
    predictor = Predictor(model, class_names, device=device)
    print(f"✓ Model loaded successfully")
    
    # Predict
    print(f"\n🔮 Running inference on {image_path.name}...")
    
    result, image_array = predictor.predict_single(str(image_path), return_image=True)
    
    # Get top-k
    probs = result['probabilities']
    top_indices = np.argsort(probs)[-args.top_k:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = probs[top_indices]
    
    # Print results
    print(f"\n{'=' * 70}")
    print(" Prediction Results")
    print(f"{'=' * 70}\n")
    
    print(f"Image: {image_path.name}")
    print(f"Primary Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    
    print(f"Top-{args.top_k} Predictions:")
    medals = ['🥇', '🥈', '🥉']
    for i, (medal, cls_name, prob) in enumerate(zip(medals, top_classes, top_probs)):
        print(f"  {medal} {cls_name:.<25} {prob:.2%}")
    
    # Save plot if requested
    if args.save_plot:
        save_path = Path(args.save_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_top_predictions(
            image_array,
            probs,
            class_names,
            top_k=args.top_k,
            save_path=str(save_path)
        )
    
    print(f"\n✓ Inference complete!")
    print()


if __name__ == '__main__':
    main()
