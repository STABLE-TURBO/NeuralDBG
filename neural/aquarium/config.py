"""
Configuration settings for Neural Aquarium IDE
"""

import os
from pathlib import Path

# Application Settings
APP_NAME = "Neural Aquarium"
APP_VERSION = "0.1.0"
DEFAULT_PORT = 8052
DEFAULT_HOST = "localhost"

# Paths
AQUARIUM_ROOT = Path(__file__).parent
PROJECT_ROOT = AQUARIUM_ROOT.parent.parent
CACHE_DIR = Path.home() / ".neural" / "aquarium"
COMPILED_SCRIPTS_DIR = CACHE_DIR / "compiled"
EXPORTED_SCRIPTS_DIR = CACHE_DIR / "exported"
TEMP_DIR = CACHE_DIR / "temp"

# Ensure directories exist
for directory in [CACHE_DIR, COMPILED_SCRIPTS_DIR, EXPORTED_SCRIPTS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Backend Configuration
SUPPORTED_BACKENDS = ["tensorflow", "pytorch", "onnx"]
DEFAULT_BACKEND = "tensorflow"

# Dataset Configuration
BUILTIN_DATASETS = {
    "MNIST": {
        "shape": (28, 28, 1),
        "classes": 10,
        "type": "image"
    },
    "CIFAR10": {
        "shape": (32, 32, 3),
        "classes": 10,
        "type": "image"
    },
    "CIFAR100": {
        "shape": (32, 32, 3),
        "classes": 100,
        "type": "image"
    },
    "ImageNet": {
        "shape": (224, 224, 3),
        "classes": 1000,
        "type": "image"
    }
}

# Training Configuration
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2
MAX_EPOCHS = 1000
MIN_EPOCHS = 1
MAX_BATCH_SIZE = 2048
MIN_BATCH_SIZE = 1

# UI Configuration
EDITOR_DEFAULT_CONTENT = """network MyModel {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}"""

CONSOLE_MAX_LINES = 10000
LOG_REFRESH_INTERVAL_MS = 500

# Theme Configuration
THEME = "darkly"
PRIMARY_COLOR = "#00BFFF"
SUCCESS_COLOR = "#28a745"
WARNING_COLOR = "#ffc107"
DANGER_COLOR = "#dc3545"
INFO_COLOR = "#17a2b8"

# Execution Configuration
PROCESS_TIMEOUT = 3600  # 1 hour max execution time
OUTPUT_BUFFER_SIZE = 1024

# Export Configuration
DEFAULT_EXPORT_DIR = "./exported_scripts"
SCRIPT_FILE_EXTENSION = ".py"
CONFIG_FILE_EXTENSION = ".json"

# Feature Flags
ENABLE_HPO = True
ENABLE_NEURALDBG = True
ENABLE_VISUALIZATION = True
ENABLE_CLOUD_EXECUTION = False  # Future feature
ENABLE_COLLABORATIVE_EDITING = False  # Future feature

# Logging
LOG_LEVEL = os.environ.get("AQUARIUM_LOG_LEVEL", "INFO")
ENABLE_DEBUG_MODE = os.environ.get("AQUARIUM_DEBUG", "false").lower() == "true"

# Environment Detection
def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_kaggle():
    """Check if running in Kaggle"""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def is_cloud_environment():
    """Check if running in any cloud environment"""
    return is_colab() or is_kaggle()

# Example Models
EXAMPLE_MODELS = {
    "MNIST CNN": """network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",
    
    "CIFAR10 CNN": """network CIFAR10CNN {
    input: (None, 32, 32, 3)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        Dropout(rate=0.25)
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        Dropout(rate=0.25)
        Flatten()
        Dense(units=512, activation=relu)
        BatchNormalization()
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",
    
    "Simple LSTM": """network TextLSTM {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64, return_sequences=True)
        LSTM(units=64)
        Dense(units=64, activation=relu)
        Dropout(rate=0.5)
        Output(units=1, activation=sigmoid)
    loss: binary_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",
    
    "Transformer": """network TransformerEncoder {
    input: (None, 512)
    layers:
        Embedding(input_dim=10000, output_dim=256)
        MultiHeadAttention(num_heads=8, key_dim=32)
        LayerNormalization()
        Dense(units=512, activation=relu)
        Dense(units=256)
        LayerNormalization()
        GlobalAveragePooling1D()
        Dense(units=64, activation=relu)
        Output(units=2, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}"""
}
