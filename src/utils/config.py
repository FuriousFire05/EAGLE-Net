"""
Configuration file for EAGLE-Net experiments.
Customize these settings for different experiments.
"""

# ============ Data Configuration ============
DATA_DIR = 'data/EuroSAT'
IMG_SIZE = 64
NUM_CLASSES = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============ Model Configuration ============
# Options: 'baseline_cnn', 'lightweight_cnn', 'eager_net'
MODEL_NAME = 'eager_net'

# ============ Training Configuration ============
SEED = 42
DEVICE = 'cuda'  # 'cuda' or 'cpu'
EPOCHS = 100
BATCH_SIZE = 64

# Learning rate and optimization
LR = 0.001
WEIGHT_DECAY = 5e-5
OPTIMIZER = 'adam'  # 'adam' or 'sgd'

# Learning rate schedule
USE_LR_SCHEDULE = False
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCH = 50

# ============ Augmentation Configuration ============
USE_AUGMENTATION = True
AUGMENTATION_TYPES = [
    'horizontal_flip',
    'vertical_flip',
    'rotation',  # 15 degrees
    'color_jitter',  # brightness, contrast, saturation
]

# ============ Checkpoint Configuration ============
SAVE_DIR = 'artifacts'
SAVE_INTERVAL = 1  # Save every N epochs
SAVE_BEST_ONLY = True

# ============ Logging Configuration ============
VERBOSE = True
PRINT_INTERVAL = 1  # Print metrics every N epochs

# ============ Inference Configuration ============
INFERENCE_BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.5
TOP_K = 3

# ============ Experiment Metadata ============
EXPERIMENT_NAME = 'EAGLE-Net v1.0'
EXPERIMENT_DESCRIPTION = (
    'Lightweight CNN with channel and spatial attention for EuroSAT classification'
)


def get_config():
    """Return configuration as dictionary."""
    return {
        'data': {
            'dir': DATA_DIR,
            'img_size': IMG_SIZE,
            'num_classes': NUM_CLASSES,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
        },
        'model': {
            'name': MODEL_NAME,
        },
        'training': {
            'seed': SEED,
            'device': DEVICE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'optimizer': OPTIMIZER,
        },
        'augmentation': {
            'enabled': USE_AUGMENTATION,
            'types': AUGMENTATION_TYPES,
        },
        'checkpoint': {
            'save_dir': SAVE_DIR,
            'save_interval': SAVE_INTERVAL,
            'save_best_only': SAVE_BEST_ONLY,
        },
        'logging': {
            'verbose': VERBOSE,
            'print_interval': PRINT_INTERVAL,
        },
    }


if __name__ == '__main__':
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
