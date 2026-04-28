# src/utils/config.py

CONFIG = {

    # ========================
    # MODEL SETTINGS
    # ========================
    "model": {
        "name": "lightweight_cnn",   # change to: baseline_cnn / lightweight_cnn / eagle_net
        "num_classes": 10
    },

    # ========================
    # DATA SETTINGS
    # ========================
    "data": {
        "root": "data/raw",
        "image_size": 64,

        "batch_size": 32,
        "num_workers": 2,

        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,

        "seed": 42
    },

    # ========================
    # TRAINING SETTINGS
    # ========================
    "training": {
        "epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,

        # scheduler
        "scheduler_step_size": 5,
        "scheduler_gamma": 0.5
    },

    # ========================
    # PATHS
    # ========================
    "paths": {
        "model_dir": "artifacts/models",
        "results_dir": "artifacts/results",
        "plots_dir": "artifacts/plots"
    }
}