# app/backend/core/model_registry.py

import os
import torch

from src.models.architectures import create_model
from src.utils.config import CONFIG


class ModelRegistry:
    """
    Loads and stores all models in memory for fast inference.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def load_all_models(self):
        """
        Load all available models into memory.
        """
        model_names = ["baseline_cnn", "lightweight_cnn", "eagle_net"]

        for name in model_names:
            print(f"[ModelRegistry] Loading model: {name}")
            self.models[name] = self._load_model(name)

        print("[ModelRegistry] All models loaded successfully")

    def _load_model(self, model_name: str):
        """
        Load a single model from disk.
        """
        num_classes = CONFIG["model"]["num_classes"]

        model = create_model(model_name, num_classes=num_classes)
        model = model.to(self.device)

        model_path = os.path.join(
            CONFIG["paths"]["model_dir"],
            f"{model_name}.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)

        model.eval()

        return model

    def get_model(self, model_name: str):
        """
        Retrieve a loaded model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found in registry: {model_name}")

        return self.models[model_name]


# Singleton instance (important)
model_registry = ModelRegistry()