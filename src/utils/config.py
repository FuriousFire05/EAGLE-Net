# src/utils/config.py

CONFIG = {
    "model": {
        "name": "baseline_cnn",  # change this only
        "num_classes": 10
    },

    "training": {
        "batch_size": 32,
        "epochs": 3,
        "learning_rate": 0.001
    },

    "paths": {
        "model_dir": "artifacts/models/"
    }
}