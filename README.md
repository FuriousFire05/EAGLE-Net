# EAGLE-Net: Efficient Attention for Geo-spatial Land Estimation Network

A lightweight CNN with channel attention mechanisms for EuroSAT RGB satellite land-use classification (10 classes, 27,000 images).

## Project Overview

**Task:** Multi-class land-use classification on EuroSAT RGB satellite imagery  
**Dataset:** EuroSAT, 10 classes, 64×64 RGB images  
**Framework:** PyTorch  
**Approach:** Lightweight CNN with CBAM/SE-style channel attention  
**Demo:** Streamlit web application  

## Directory Structure

```
EAGLE-Net/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/                    # Dataset directory (raw/processed)
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── data/               # Dataset loading and preprocessing
│   ├── models/             # Model definitions (BaselineCNN, LightweightCNN, EAGLENet)
│   ├── training/           # Training logic and utilities
│   ├── inference/          # Inference and prediction functions
│   └── utils/              # Helper utilities (metrics, visualization, etc.)
├── app/                    # Streamlit demo application
├── artifacts/              # Saved models and training artifacts
└── report/                 # Results and analysis reports
```

## Installation

1. **Clone or download the repository:**
   ```bash
   cd EAGLE-Net
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Download Dataset
```bash
python src/data/download_eurosat.py
```

### 2. Train Model
```bash
python train.py --model eager_net --epochs 100 --batch-size 64 --lr 0.001
```

Available models: `baseline_cnn`, `lightweight_cnn`, `eager_net`

### 3. Run Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

## Features

### Models

- **BaselineCNN:** Standard CNN with 4 conv blocks (baseline)
- **LightweightCNN:** Efficient depthwise separable convolutions
- **EAGLENet:** LightweightCNN + Channel Attention Mechanism (CBAM-style)

### Training

- Multi-class classification (10 EuroSAT classes)
- Reproducible train/val/test split
- Metrics: Accuracy, Macro F1, Precision, Recall
- Confusion matrix visualization
- Model checkpointing

### Inference

- Single image prediction
- Top-3 class probabilities
- Batch inference support

### Visualization

- Training/validation curves
- Confusion matrix heatmap
- Class distribution plots
- Streamlit interactive demo

## Training Configuration

Default hyperparameters can be modified via command-line arguments:

```bash
python train.py \
    --model eager_net \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 5e-5 \
    --seed 42
```

## Model Architecture

### EAGLENet (Efficient + Attention)

1. **Lightweight Backbone:**
   - Depthwise separable convolutions for efficiency
   - Reduced parameters vs. standard CNN

2. **Channel Attention (CBAM-style):**
   - Channel attention block (squeeze + excitation)
   - Spatial attention block (adaptive weighting)
   - Applied after residual blocks

3. **Classification Head:**
   - Global average pooling
   - Fully connected layer for 10 classes

## Metrics and Evaluation

After training, metrics are saved to `artifacts/`:
- Model checkpoint: `best_model.pt`
- Training curves plot: `training_curves.png`
- Confusion matrix: `confusion_matrix.png`
- Metrics summary: `metrics.json`

## Streamlit Demo

Launch the interactive app:

```bash
streamlit run app/streamlit_app.py
```

Features:
- Upload EuroSAT RGB images (64×64)
- Real-time predictions
- Top-3 class probabilities
- Confidence scores
- Class distribution visualization

## File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `src/data/dataset.py` | EuroSAT dataset loader |
| `src/data/download_eurosat.py` | Dataset download utility |
| `src/models/architectures.py` | Model definitions |
| `src/training/trainer.py` | Training loop and validation |
| `src/inference/predictor.py` | Inference and prediction functions |
| `src/utils/metrics.py` | Metric computation |
| `src/utils/visualization.py` | Plotting utilities |
| `app/streamlit_app.py` | Streamlit demo application |

## Notes for Students

- **Reproducibility:** Set `--seed` to control randomness
- **Debugging:** Use `--debug` flag for verbose logging
- **Validation:** Check `artifacts/confusion_matrix.png` to inspect per-class performance
- **Efficiency:** LightweightCNN uses ~40% fewer parameters than BaselineCNN
- **Attention:** EAGLENet channel attention helps refine spatial relevance

## Dependencies

- PyTorch 2.0.1
- torchvision 0.15.2
- scikit-learn (metrics)
- matplotlib, seaborn (visualization)
- Streamlit (web app)
- NumPy, Pandas

## License

Educational project for deep learning assignment.

## Author

EAGLE-Net Assignment Project
