"""
Model Trainer for Neural models.

This module provides functionality to train models and collect metrics during training.
"""

import os
import sys
import tempfile
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains models and collects metrics during training."""

    def __init__(self, model_data: Dict[str, Any], trace_data: List[Dict[str, Any]], backend: str = 'tensorflow') -> None:
        """
        Initialize the model trainer.

        Args:
            model_data: The model data from the parser
            trace_data: The trace data to update with real metrics
            backend: The backend framework ('tensorflow' or 'pytorch')
        """
        self.model_data = model_data
        self.trace_data = trace_data
        self.backend = backend

    def train_tensorflow_model(self, dataset: str = 'mnist') -> Tuple[Any, Any, Any]:
        """
        Train a TensorFlow model and collect metrics.

        Args:
            dataset: The dataset to use ('mnist' or 'cifar10')

        Returns:
            Tuple of (model, x_train, y_train)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.datasets import mnist, cifar10
            from tensorflow.keras.utils import to_categorical

            # Load dataset
            if dataset.lower() == 'mnist':
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                # Reshape and normalize
                x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
                x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            elif dataset.lower() == 'cifar10':
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                # Normalize
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0
            else:
                logger.warning(f"Dataset {dataset} not supported. Using MNIST.")
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
                x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

            # One-hot encode labels
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            # Create the model
            model = self._create_tensorflow_model()

            return model, x_train, y_train

        except Exception as e:
            logger.error(f"Failed to train TensorFlow model: {str(e)}")
            return None, None, None

    def train_pytorch_model(self, dataset: str = 'mnist') -> Tuple[Any, Any, Any, Any]:
        """
        Train a PyTorch model and collect metrics.

        Args:
            dataset: The dataset to use ('mnist' or 'cifar10')

        Returns:
            Tuple of (model, data_loader, criterion, optimizer)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
            import torchvision.datasets as datasets
            import torchvision.transforms as transforms

            # Load dataset
            if dataset.lower() == 'mnist':
                # Load MNIST dataset
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            elif dataset.lower() == 'cifar10':
                # Load CIFAR10 dataset
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            else:
                logger.warning(f"Dataset {dataset} not supported. Using MNIST.")
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Create the model
            model = self._create_pytorch_model()

            # Set up optimizer
            optimizer_config = self.model_data.get('optimizer', {'type': 'Adam', 'params': {'lr': 0.001}})
            optimizer_type = optimizer_config.get('type', 'Adam')
            optimizer_params = optimizer_config.get('params', {'lr': 0.001})

            if optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), **optimizer_params)
            elif optimizer_type == 'SGD':
                optimizer = optim.SGD(model.parameters(), **optimizer_params)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Loss function
            criterion = nn.CrossEntropyLoss()

            return model, train_loader, criterion, optimizer

        except Exception as e:
            logger.error(f"Failed to train PyTorch model: {str(e)}")
            return None, None, None, None

    def _create_tensorflow_model(self) -> Any:
        """Create a TensorFlow model from the model data."""
        try:
            import tensorflow as tf
            from neural.code_generation.code_generator import generate_code

            # Generate TensorFlow code
            model_code = generate_code(self.model_data, 'tensorflow')

            # Create a temporary file for the model code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
                f.write(model_code.encode())
                model_file = f.name

            # Import the model
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_module", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            # Get the model
            model = model_module.model

            # Compile the model
            optimizer_config = self.model_data.get('optimizer', {'type': 'Adam', 'params': {'learning_rate': 0.001}})
            optimizer_type = optimizer_config.get('type', 'Adam')
            optimizer_params = optimizer_config.get('params', {'learning_rate': 0.001})

            if optimizer_type == 'Adam':
                optimizer = tf.keras.optimizers.Adam(**optimizer_params)
            elif optimizer_type == 'SGD':
                optimizer = tf.keras.optimizers.SGD(**optimizer_params)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            logger.error(f"Failed to create TensorFlow model: {str(e)}")
            return None

    def _create_pytorch_model(self) -> Any:
        """Create a PyTorch model from the model data."""
        try:
            import torch
            from neural.code_generation.code_generator import generate_code

            # Generate PyTorch code
            model_code = generate_code(self.model_data, 'pytorch')

            # Create a temporary file for the model code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
                f.write(model_code.encode())
                model_file = f.name

            # Import the model
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_module", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            # Get the model
            model = model_module.model

            return model

        except Exception as e:
            logger.error(f"Failed to create PyTorch model: {str(e)}")
            return None
