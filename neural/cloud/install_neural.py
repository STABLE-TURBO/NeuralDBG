"""
Neural DSL Cloud Installation Script
This script installs Neural DSL in cloud environments like Kaggle and Google Colab.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_step(message):
    """Print a step message with formatting."""
    print(f"\n\033[1;34m>> {message}\033[0m")

def print_success(message):
    """Print a success message with formatting."""
    print(f"\033[1;32m✓ {message}\033[0m")

def print_error(message):
    """Print an error message with formatting."""
    print(f"\033[1;31m✗ {message}\033[0m")

def detect_environment():
    """Detect the cloud environment we're running in."""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    return 'unknown'

def install_neural():
    """Install Neural DSL from GitHub."""
    print_step("Installing Neural DSL...")

    try:
        # Clone the repository
        subprocess.check_call([
            "git", "clone", "https://github.com/Lemniscate-SHA-256/Neural.git"
        ])

        # Install the package
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "./Neural"
        ])

        print_success("Neural DSL installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Neural DSL: {e}")
        return False

def setup_environment(env_type):
    """Set up the environment for Neural DSL."""
    print_step(f"Setting up environment for {env_type}...")

    # Create a directory for Neural models
    models_dir = Path("neural_models")
    models_dir.mkdir(exist_ok=True)

    # Set environment variables
    os.environ['NEURAL_MODELS_DIR'] = str(models_dir.absolute())

    # Environment-specific setup
    if env_type == 'kaggle':
        # Kaggle-specific setup
        os.environ['NEURAL_FORCE_CPU'] = '0'  # Use GPU if available
    elif env_type == 'colab':
        # Colab-specific setup
        os.environ['NEURAL_FORCE_CPU'] = '0'  # Use GPU if available

    print_success(f"Environment set up for {env_type}!")
    return True

def create_example_model():
    """Create an example Neural DSL model."""
    print_step("Creating example model...")

    example_model = """
network MnistCNN {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
"""

    model_path = Path("neural_models/mnist_cnn.neural")
    model_path.write_text(example_model)

    print_success(f"Example model created at {model_path}")
    return model_path

def main():
    """Main function."""
    print_step("Neural DSL Cloud Setup")

    # Detect environment
    env_type = detect_environment()
    print(f"Detected environment: {env_type}")

    # Install Neural DSL
    if not install_neural():
        return

    # Set up environment
    if not setup_environment(env_type):
        return

    # Create example model
    model_path = create_example_model()

    # Print usage instructions
    print_step("Neural DSL is ready to use!")
    print("You can now use Neural DSL commands, for example:")
    print("\nCompile the example model:")
    print(f"!python -m neural.cli compile {model_path} --backend tensorflow")

    print("\nVisualize the model:")
    print(f"!python -m neural.cli visualize {model_path}")

    print("\nDebug the model (will start a dashboard):")
    print(f"!python -m neural.cli debug {model_path}")

    print("\nFor more information, see the Neural DSL documentation:")
    print("https://github.com/Lemniscate-SHA-256/Neural")

if __name__ == "__main__":
    main()
