# src/utils/config.py
"""Central experiment configuration for model, data, training, and output paths."""

CONFIG = {
    # ========================
    # MODEL SETTINGS
    # ========================
    "model": {
        "name": "eagle_net",   # baseline_cnn / lightweight_cnn / eagle_net
        "num_classes": 10,
    },

    # ========================
    # DATA SETTINGS
    # ========================
    "data": {
        # EuroSAT data root and square image size used by all model variants.
        "root": "data/raw",
        "image_size": 64,

        # DataLoader runtime settings.
        "batch_size": 32,
        "num_workers": 2,

        # Dataset split proportions. The remaining examples after train/val
        # allocation are assigned to the test split in the dataloader.
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,

        "seed": 42,
    },

    # ========================
    # TRAINING SETTINGS
    # ========================
    "training": {
        "epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,

        "scheduler_step_size": 5,
        "scheduler_gamma": 0.5,

        # Robust training is intended mainly for eagle_net.
        "robust_training": True,
    },

    # ========================
    # PATHS
    # ========================
    "paths": {
        # Output directories for checkpoints, evaluation results, and figures.
        "model_dir": "artifacts/models",
        "results_dir": "artifacts/results",
        "plots_dir": "artifacts/plots",
    },
}
