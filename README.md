# EAGLE-Net: Robust Satellite Image Classification

**A PyTorch project exploring how satellite image classifiers behave under real-world visual distortions.**

EAGLE-Net compares a standard CNN baseline, a lightweight low-latency model, and a custom robustness-oriented architecture on the EuroSAT dataset. The project focuses less on chasing a single headline score and more on understanding how architecture choices affect robustness, latency, and failure modes across different image conditions.

## Project Overview

Satellite image classification models are often evaluated on clean benchmark data, but deployed imagery can be affected by noise, blur, compression, lighting changes, resolution loss, and class-specific ambiguity. This project studies those conditions directly by training and evaluating multiple CNN architectures under a shared experimental setup.

The main model, **EAGLE-Net**, is designed to improve robustness through multiscale feature extraction, channel attention, spatial gating, and anti-aliased downsampling. Its performance is compared against a larger baseline CNN and a compact depthwise-separable CNN.

## Key Features

- Robustness-aware training pipeline for satellite imagery.
- Three model families with different accuracy and efficiency tradeoffs.
- Evaluation across clean data and multiple simulated deployment distortions.
- Saved JSON results for reproducible analysis.
- Generated comparison plots for accuracy, F1 score, and latency tradeoffs.
- Presentation-ready analysis notebook for reviewing findings.

## Model Architectures

### BaselineCNN

A conventional convolutional neural network with stacked convolution, batch normalization, ReLU, and max-pooling blocks. It serves as a strong reference model for clean and standard evaluation settings.

### LightweightCNN

An efficient CNN built with depthwise separable convolutions. This model is optimized for lower inference latency and a smaller parameter footprint, making it useful as a deployment-oriented comparison point.

### EAGLE-Net

A custom robustness-focused architecture built around:

- Dual-kernel inverted residual blocks for multiscale spatial features.
- Squeeze-and-excitation attention for channel recalibration.
- Spatial gating to emphasize informative image regions.
- BlurPool downsampling to reduce aliasing during spatial resolution changes.

## Experimental Setup

**Dataset:** EuroSAT  
**Task:** Multiclass satellite image classification  
**Input size:** 64 x 64 RGB images  
**Framework:** PyTorch  

The models are evaluated across the following conditions:

- `clean`
- `noisy`
- `low_light`
- `blurred`
- `hard_subset`
- `jpeg`
- `color_shift`
- `strong_noise`
- `downscale`

Results are saved under `artifacts/results/` as JSON files, with one multi-condition report per model.

## Results Summary

- **EAGLE-Net performs best under most distortions**, suggesting that multiscale blocks, attention, and anti-aliased downsampling improve robustness across several corruption types.
- **BaselineCNN performs better under JPEG compression**, showing that robustness is not universal and can depend strongly on the specific distortion.
- **LightweightCNN offers the best latency**, but its compact design comes with lower accuracy under several evaluation conditions.
- **Robustness varies across corruption types**, so a model that is strong under noise or blur may not be the strongest under compression or resolution loss.

**Note:** Hard subset metrics are computed on a filtered set of classes and are not directly comparable to full-dataset metrics.

## Plots

### Accuracy Across Conditions

![Accuracy comparison](artifacts/plots/accuracy_comparison.png)

### F1 Score Across Conditions

![F1 comparison](artifacts/plots/f1_comparison.png)

### Latency vs Accuracy Tradeoff

![Latency tradeoff](artifacts/plots/latency_tradeoff.png)

## Key Insights

The central takeaway is that robustness is distribution-dependent. Architectural choices affect how models respond to different distortions, but no design should be assumed to dominate every deployment condition.

EAGLE-Net provides the strongest overall robustness profile in these experiments, while the BaselineCNN and LightweightCNN highlight two important practical tradeoffs: corruption-specific resilience and inference speed. This makes the project useful not only as a model comparison, but also as a reminder that evaluation should match the conditions a model is likely to face after deployment.

## Folder Structure

```text
EAGLE-Net/
├── app/                     # Application or demo code
├── artifacts/
│   ├── models/              # Saved model checkpoints and training histories
│   ├── plots/               # Generated comparison plots
│   └── results/             # Evaluation results in JSON format
├── data/                    # Local dataset storage
├── notebooks/
│   ├── eagle_net_analysis.ipynb
│   └── plot_results.py      # Plot generation script
├── report/                  # Report assets or writeups
├── src/
│   ├── data/                # Dataloaders, training transforms, eval corruptions
│   ├── models/              # BaselineCNN, LightweightCNN, and EAGLE-Net
│   ├── training/            # Training and evaluation entry points
│   └── utils/               # Shared configuration
├── requirements.txt
└── README.md
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model selected in `src/utils/config.py`:

```bash
python -m src.training.train_model
```

Evaluate the trained model across all conditions:

```bash
python -m src.training.evaluate_model
```

Generate result plots:

```bash
python notebooks/plot_results.py
```

Open the analysis notebook:

```bash
jupyter notebook notebooks/eagle_net_analysis.ipynb
```

To train or evaluate a different architecture, update `CONFIG["model"]["name"]` in `src/utils/config.py` to one of:

- `baseline_cnn`
- `lightweight_cnn`
- `eagle_net`

## Future Work

- Add calibration analysis to understand confidence under distribution shift.
- Evaluate on additional satellite datasets beyond EuroSAT.
- Test more realistic sensor and atmospheric corruptions.
- Add class-level robustness summaries for each distortion type.
- Explore model compression or quantization for EAGLE-Net.
- Expand the analysis notebook with tables generated directly from JSON results.

## Limitations

- The evaluation is based on synthetic, corruption-based testing rather than cross-dataset validation. While this helps simulate deployment conditions, it does not fully capture real-world distribution shifts across different sensors or geographies.
- Some distribution-shifted corruptions (e.g., strong noise and color shift) are related to training-time augmentations, so results should be interpreted as robustness under shifted rather than completely unseen conditions.
- Hard subset metrics are computed on a filtered class distribution and are not directly comparable to full-dataset metrics.
- The study focuses on classification performance and latency, and does not evaluate model calibration or confidence under distribution shift.

## 👨‍💻 Author

Created by [FuriousFire](https://github.com/FuriousFire05)