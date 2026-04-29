# EAGLE-Net: Robust Satellite Image Classification

**Understanding how CNN architectures behave under real-world visual distortions.**

EAGLE-Net is a PyTorch-based project that studies how different convolutional neural network architectures perform under realistic image degradations such as noise, blur, compression, lighting variation, and resolution loss.

Instead of optimizing for a single benchmark score, this project focuses on **robustness, failure modes, and deployment-aware tradeoffs** across multiple model designs.

---

## Project Overview

Satellite image classification models are typically evaluated on clean benchmark datasets. However, real-world imagery often contains noise, compression artifacts, lighting shifts, and resolution degradation.

This project investigates how different CNN architectures behave under such conditions by training and evaluating multiple models under a shared experimental framework.

The primary model, **EAGLE-Net**, is designed to improve robustness using multiscale feature extraction, attention mechanisms, and anti-aliased downsampling. Its performance is compared against:

- A standard **BaselineCNN**
- A latency-optimized **LightweightCNN**

---

## Why This Matters

Most machine learning models are evaluated under ideal conditions, but deployed systems rarely encounter clean inputs.

Understanding how models respond to distribution shifts is critical for:

- Reliable real-world deployment
- Failure-mode analysis
- Architecture design decisions

This project emphasizes **how models behave**, not just how they score.

---

## Key Features

- Robustness-aware training pipeline for satellite imagery
- Three model families with distinct design tradeoffs
- Evaluation across clean and distribution-shifted conditions
- JSON-based result storage for reproducibility
- Comparison plots for accuracy, F1 score, and latency
- Presentation-ready analysis notebook

---

## Model Architectures

### BaselineCNN

A conventional convolutional neural network using stacked convolution, batch normalization, ReLU, and max-pooling layers.

Serves as a strong reference model for standard evaluation.

---

### LightweightCNN

A compact CNN using depthwise separable convolutions.

Optimized for:

- Lower latency
- Reduced parameter count

Provides a deployment-focused comparison point.

---

### EAGLE-Net

A custom robustness-focused architecture incorporating:

- **Dual-kernel inverted residual blocks** for multiscale spatial features
- **Squeeze-and-excitation attention** for channel recalibration
- **Spatial gating** to emphasize informative regions
- **BlurPool downsampling** to reduce aliasing effects

---

## Experimental Setup

- **Dataset:** EuroSAT
- **Task:** Multiclass satellite image classification
- **Input Size:** 64 × 64 RGB images
- **Framework:** PyTorch

Models are evaluated across the following conditions:

- `clean`
- `noisy`
- `low_light`
- `blurred`
- `hard_subset`
- `jpeg`
- `color_shift`
- `strong_noise`
- `downscale`

Results are stored as JSON files in:

```text
artifacts/results/
```

---

## Results Summary

- **EAGLE-Net performs best under most distortions**, indicating that multiscale features, attention, and anti-aliased downsampling improve robustness across several conditions.
- **BaselineCNN performs better under JPEG compression**, showing that robustness is not universal and depends on the distortion type.
- **LightweightCNN achieves the lowest latency**, but sacrifices accuracy under several conditions.
- **Robustness varies across corruption types**, meaning strong performance under one distortion does not guarantee performance under others.

> **Key Insight:** Robustness is distribution-dependent — different architectures specialize in different types of visual distortions.

**Note:** Hard subset metrics are computed on a filtered set of classes and are not directly comparable to full-dataset metrics.

---

## Plots

### Accuracy Across Conditions

![Accuracy comparison](artifacts/plots/accuracy_comparison.png)

### F1 Score Across Conditions

![F1 comparison](artifacts/plots/f1_comparison.png)

### Latency vs Accuracy Tradeoff

![Latency tradeoff](artifacts/plots/latency_tradeoff.png)

---

## Key Insights

The central takeaway is that **no single architecture is universally robust**.

- EAGLE-Net provides the strongest overall robustness profile
- BaselineCNN remains competitive under compression artifacts
- LightweightCNN highlights efficiency vs accuracy tradeoffs

This makes the project valuable not only for model comparison, but also for understanding **how architectural bias influences robustness behavior**.

---

## Folder Structure

```text
EAGLE-Net/
├── artifacts/
│   ├── plots/
│   └── results/
├── notebooks/
│   ├── eagle_net_analysis.ipynb
│   └── plot_results.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## How to Run

### Create virtual environment

```bash
python -m venv .venv
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python -m src.training.train_model
```

### Evaluate model

```bash
python -m src.training.evaluate_model
```

### Generate plots

```bash
python notebooks/plot_results.py
```

### Open notebook

```bash
jupyter notebook notebooks/eagle_net_analysis.ipynb
```

To switch models, update:

```python
CONFIG["model"]["name"]
```

Options:

- `baseline_cnn`
- `lightweight_cnn`
- `eagle_net`

---

## Limitations

- Evaluation is based on synthetic corruption-based testing rather than cross-dataset validation.
- Some distribution-shifted corruptions are related to training-time augmentations.
- Hard subset metrics are not directly comparable to full-dataset metrics.
- Model calibration under distribution shift is not evaluated.

---

## Future Work

- Cross-dataset generalization testing
- Model calibration analysis
- More realistic sensor-level corruptions
- Class-level robustness breakdowns
- Model compression for EAGLE-Net

---

## 👨‍💻 Author

Created by [FuriousFire](https://github.com/FuriousFire05)

---

## License

This project is licensed under the MIT License.
