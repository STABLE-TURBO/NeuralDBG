from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor

from neural.exceptions import (
    HPOException,
    InvalidHPOConfigError,
    InvalidParameterError,
    UnsupportedBackendError,
)
from neural.execution_optimization.execution import get_device
from neural.parser.parser import ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator


def validate_hpo_categorical(param_name: str, values: List[Any]) -> List[Any]:
    """
    Validate categorical HPO parameter values.
    
    Args:
        param_name: Name of the parameter being validated
        values: List of categorical values to validate
        
    Returns:
        Validated list of values
        
    Raises:
        InvalidHPOConfigError: If values are invalid
    """
    if not isinstance(values, list):
        raise InvalidHPOConfigError(
            f"Categorical values for {param_name} must be a list, "
            f"got {type(values).__name__}"
        )
    
    if len(values) == 0:
        raise InvalidHPOConfigError(
            f"Categorical values for {param_name} cannot be empty"
        )
    
    if len(values) > 100:
        raise InvalidHPOConfigError(
            f"Too many categorical values for {param_name} (max 100): "
            f"{len(values)}"
        )
    
    # Check for type consistency
    first_type = type(values[0])
    if not all(isinstance(v, first_type) for v in values):
        raise InvalidHPOConfigError(
            f"Categorical values for {param_name} must all be the same type, got mixed types: "
            f"{set(type(v).__name__ for v in values)}"
        )
    
    return values


def validate_hpo_bounds(
    param_name: str, low: float, high: float, hpo_type: str
) -> Tuple[float, float]:
    """
    Validate bounds for range or log_range HPO parameters.
    
    Args:
        param_name: Name of the parameter being validated
        low: Lower bound
        high: Upper bound
        hpo_type: Type of HPO ('range' or 'log_range')
        
    Returns:
        Tuple of validated (low, high) bounds
        
    Raises:
        InvalidHPOConfigError: If bounds are invalid
    """
    if low is None or high is None:
        raise InvalidHPOConfigError(
            f"Both low and high bounds must be specified for "
            f"{param_name} {hpo_type}"
        )
    
    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        raise InvalidHPOConfigError(
            f"Bounds for {param_name} must be numeric, "
            f"got low={low}, high={high}"
        )
    
    if low >= high:
        raise InvalidHPOConfigError(
            f"Lower bound must be less than upper bound for {param_name}, "
            f"got low={low}, high={high}"
        )
    
    if hpo_type == 'log_range':
        if low <= 0:
            raise InvalidHPOConfigError(
                f"log_range for {param_name} requires positive bounds, got low={low}"
            )
        if high <= 0:
            raise InvalidHPOConfigError(
                f"log_range for {param_name} requires positive bounds, got high={high}"
            )
    
    return low, high


# Data Loader
def get_data(
    dataset_name: str, 
    input_shape: Tuple[int, ...], 
    batch_size: int, 
    train: bool = True, 
    backend: str = 'pytorch'
) -> Union[torch.utils.data.DataLoader, tf.data.Dataset]:
    """
    Get a data loader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('MNIST' or 'CIFAR10')
        input_shape: Shape of input tensors
        batch_size: Batch size for data loading
        train: Whether to load training or validation data
        backend: Backend to use ('pytorch' or 'tensorflow')
        
    Returns:
        Data loader for the specified backend
        
    Raises:
        ValueError: If dataset or backend is unsupported
    """
    datasets = {'MNIST': MNIST, 'CIFAR10': CIFAR10}
    if dataset_name not in datasets:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Available: {list(datasets.keys())}"
        )
    
    dataset = datasets[dataset_name](
        root='./data', train=train, transform=ToTensor(), download=True
    )
    
    if backend == 'pytorch':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    elif backend == 'tensorflow':
        data = dataset.data.numpy() / 255.0
        targets = dataset.targets.numpy()
        if len(data.shape) == 3:
            data = data[..., None]
        return tf.data.Dataset.from_tensor_slices((data, targets)).batch(batch_size)
    else:
        raise UnsupportedBackendError(backend=backend, available_backends=['pytorch', 'tensorflow'])


def prod(iterable: Tuple[int, ...]) -> int:
    """Calculate the product of all elements in a tuple."""
    result = 1
    for x in iterable:
        result *= x
    return result


# Factory Function
def create_dynamic_model(
    model_dict: Dict[str, Any], 
    trial: Trial, 
    hpo_params: List[Dict[str, Any]], 
    backend: str = 'pytorch'
) -> Union['DynamicPTModel', 'DynamicTFModel']:
    """
    Create a dynamic model with HPO parameters resolved by the trial.
    
    Args:
        model_dict: Model configuration dictionary
        trial: Optuna trial for suggesting hyperparameters
        hpo_params: List of HPO parameter specifications
        backend: Backend to use ('pytorch' or 'tensorflow')
        
    Returns:
        Dynamic model instance for the specified backend
        
    Raises:
        UnsupportedBackendError: If backend is not supported
        InvalidHPOConfigError: If HPO configuration is invalid
    """
    resolved_model_dict = copy.deepcopy(model_dict)

    # Resolve HPO parameters in layers
    for layer in resolved_model_dict['layers']:
        if 'params' in layer and layer['params']:
            for param_name, param_value in layer['params'].items():
                if isinstance(param_value, dict) and 'hpo' in param_value:
                    hpo = param_value['hpo']
                    if hpo['type'] == 'categorical':
                        values = validate_hpo_categorical(param_name, hpo['values'])
                        layer['params'][param_name] = trial.suggest_categorical(
                            f"{layer['type']}_{param_name}", values
                        )
                    elif hpo['type'] == 'range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'range')
                        step = hpo.get('step')
                        if step and step is not False:
                            layer['params'][param_name] = trial.suggest_float(
                                f"{layer['type']}_{param_name}",
                                low,
                                high,
                                step=step
                            )
                        else:
                            layer['params'][param_name] = trial.suggest_float(
                                f"{layer['type']}_{param_name}",
                                low,
                                high
                            )
                    elif hpo['type'] == 'log_range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'log_range')
                        layer['params'][param_name] = trial.suggest_float(
                            f"{layer['type']}_{param_name}",
                            low,
                            high,
                            log=True
                        )

    # Resolve HPO parameters in optimizer
    if 'optimizer' in resolved_model_dict and resolved_model_dict['optimizer']:
        optimizer = resolved_model_dict['optimizer']
        if 'params' in optimizer and optimizer['params']:
            for param_name, param_value in resolved_model_dict['optimizer']['params'].items():
                if isinstance(param_value, dict) and 'hpo' in param_value:
                    hpo = param_value['hpo']
                    opt_params = resolved_model_dict['optimizer']['params']
                    if hpo['type'] == 'categorical':
                        values = validate_hpo_categorical(param_name, hpo['values'])
                        opt_params[param_name] = trial.suggest_categorical(
                            f"opt_{param_name}", values
                        )
                    elif hpo['type'] == 'range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'range')
                        step = hpo.get('step')
                        if step and step is not False:
                            opt_params[param_name] = trial.suggest_float(
                                f"opt_{param_name}", low, high, step=step
                            )
                        else:
                            opt_params[param_name] = trial.suggest_float(
                                f"opt_{param_name}", low, high
                            )
                    elif hpo['type'] == 'log_range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'log_range')
                        opt_params[param_name] = trial.suggest_float(
                            f"opt_{param_name}", low, high, log=True
                        )

    if backend == 'pytorch':
        return DynamicPTModel(resolved_model_dict, trial, hpo_params)
    elif backend == 'tensorflow':
        return DynamicTFModel(resolved_model_dict, trial, hpo_params)
    else:
        raise UnsupportedBackendError(
            backend=backend,
            available_backends=['pytorch', 'tensorflow']
        )


def resolve_hpo_params(
    model_dict: Dict[str, Any], 
    trial: Trial, 
    hpo_params: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Resolve HPO parameters in model dictionary using Optuna trial.
    
    Args:
        model_dict: Model configuration dictionary
        trial: Optuna trial for suggesting hyperparameters
        hpo_params: List of HPO parameter specifications
        
    Returns:
        Model dictionary with HPO parameters resolved to concrete values
    """
    import copy
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    
    resolved_dict = copy.deepcopy(model_dict)

    # Resolve HPO parameters in layers
    for i, layer in enumerate(resolved_dict['layers']):
        if 'params' in layer and layer['params'] is not None:
            for param_name, param_value in layer['params'].items():
                if isinstance(param_value, dict) and 'hpo' in param_value:
                    hpo = param_value['hpo']
                    key = f"{layer['type']}_{param_name}_{i}"
                    
                    if hpo['type'] == 'categorical':
                        values = validate_hpo_categorical(param_name, hpo['values'])
                        layer['params'][param_name] = trial.suggest_categorical(key, values)
                    elif hpo['type'] == 'range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'range')
                        step = hpo.get('step')
                        if step and step is not False:
                            layer['params'][param_name] = trial.suggest_float(
                                key, low, high, step=step
                            )
                        else:
                            layer['params'][param_name] = trial.suggest_float(key, low, high)
                    elif hpo['type'] == 'log_range':
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        low, high = validate_hpo_bounds(param_name, low, high, 'log_range')
                        layer['params'][param_name] = trial.suggest_float(key, low, high, log=True)

    # Resolve HPO parameters in optimizer
    if resolved_dict.get('optimizer') and 'params' in resolved_dict['optimizer']:
        # Clean up optimizer type
        opt_type = resolved_dict['optimizer']['type']
        if '(' in opt_type:
            resolved_dict['optimizer']['type'] = opt_type[:opt_type.index('(')].capitalize()

        for param, val in resolved_dict['optimizer']['params'].items():
            if isinstance(val, dict) and 'hpo' in val:
                hpo = val['hpo']
                if hpo['type'] == 'categorical':
                    values = validate_hpo_categorical(param, hpo['values'])
                    resolved_dict['optimizer']['params'][param] = trial.suggest_categorical(
                        f"opt_{param}", values
                    )
                elif hpo['type'] == 'range':
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    low, high = validate_hpo_bounds(param, low, high, 'range')
                    step = hpo.get('step')
                    if step and step is not False:
                        resolved_dict['optimizer']['params'][param] = trial.suggest_float(
                            f"opt_{param}", low, high, step=step
                        )
                    else:
                        resolved_dict['optimizer']['params'][param] = trial.suggest_float(
                            f"opt_{param}", low, high
                        )
                elif hpo['type'] == 'log_range':
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    low, high = validate_hpo_bounds(param, low, high, 'log_range')
                    resolved_dict['optimizer']['params'][param] = trial.suggest_float(
                        f"opt_{param}", low, high, log=True
                    )

    return resolved_dict


# Dynamic Models
class DynamicPTModel(nn.Module):
    """
    Dynamic PyTorch model that builds layers based on configuration.
    
    Supports HPO by allowing trial-based parameter suggestion during construction.
    """
    def __init__(
        self, 
        model_dict: Dict[str, Any], 
        trial: Trial, 
        hpo_params: List[Dict[str, Any]]
    ) -> None:
        super().__init__()
        self.model_dict: Dict[str, Any] = model_dict
        self.layers: nn.ModuleList = nn.ModuleList()
        self.shape_propagator: ShapePropagator = ShapePropagator(debug=False)
        input_shape_raw = model_dict['input']['shape']
        input_shape = (None, input_shape_raw[-1], *input_shape_raw[:-1])
        current_shape = input_shape
        in_channels = input_shape[1]
        in_features = None

        for layer in model_dict['layers']:
            params = layer['params'] if layer['params'] is not None else {}
            params = params.copy()

            # Compute in_features from current shape before propagation
            if layer['type'] in ['Dense', 'Output'] and in_features is None:
                in_features = prod(current_shape[1:])
                self.layers.append(nn.Flatten())

            # Propagate shape after setting in_features
            current_shape = self.shape_propagator.propagate(
                current_shape, layer, framework='pytorch'
            )

            if layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int('conv_filters', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                in_channels = filters
            elif layer['type'] == 'MaxPooling2D':
                pool_size = params.get('pool_size', trial.suggest_int('maxpool2d_pool_size', 2, 3))
                stride = params.get('stride', pool_size)
                self.layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=stride))
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = prod(current_shape[1:])
            elif layer['type'] == 'Dense':
                units = params.get('units')
                if units is None:
                    units = trial.suggest_int('dense_units', 64, 256)
                if in_features is None or in_features <= 0:
                    raise InvalidParameterError(
                        parameter='in_features',
                        value=in_features,
                        layer_type='Dense',
                        expected='positive integer'
                    )
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    rate = trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)
                self.layers.append(nn.Dropout(p=rate))
            elif layer['type'] == 'Output':
                units = params.get('units', 10)
                if in_features is None or in_features <= 0:
                    raise InvalidParameterError(
                        parameter='in_features',
                        value=in_features,
                        layer_type='Output',
                        expected='positive integer'
                    )
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'LSTM':
                input_size = current_shape[-1] if len(current_shape) > 1 else in_features
                units = params.get('units', trial.suggest_int('lstm_units', 32, 256))
                num_layers = params.get('num_layers', 1)
                num_layers_param = params.get('num_layers')
                if isinstance(num_layers_param, dict) and 'hpo' in num_layers_param:
                    num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
                self.layers.append(
                    nn.LSTM(input_size, units, num_layers=num_layers, batch_first=True)
                )
                in_features = units
            elif layer['type'] == 'BatchNormalization':
                # momentum parameter can be suggested but BatchNorm2d uses default
                self.layers.append(nn.BatchNorm2d(in_channels))
            elif layer['type'] == 'Transformer':
                d_model = params.get(
                    'd_model', trial.suggest_int('transformer_d_model', 64, 512)
                )
                nhead = params.get('nhead', trial.suggest_int('transformer_nhead', 4, 8))
                num_encoder_layers = params.get(
                    'num_encoder_layers',
                    trial.suggest_int('transformer_encoder_layers', 1, 4)
                )
                num_decoder_layers = params.get(
                    'num_decoder_layers',
                    trial.suggest_int('transformer_decoder_layers', 1, 4)
                )
                dim_feedforward = params.get(
                    'dim_feedforward',
                    trial.suggest_int('transformer_ff_dim', 128, 1024)
                )
                self.layers.append(nn.Transformer(d_model=d_model,
                                                  nhead=nhead,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers,
                                                  dim_feedforward=dim_feedforward))
                in_features = d_model
            else:
                raise ValueError(f"Unsupported layer type: {layer['type']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.Transformer):
                x = layer(x, x)
            else:
                x = layer(x)
        return x


class DynamicTFModel(tf.keras.Model):
    """
    Dynamic TensorFlow model that builds layers based on configuration.
    
    Supports HPO by allowing trial-based parameter suggestion during construction.
    """
    def __init__(
        self, 
        model_dict: Dict[str, Any], 
        trial: Trial, 
        hpo_params: List[Dict[str, Any]]
    ) -> None:
        super().__init__()
        self.model_dict = model_dict
        self.layers_list: List[Any] = []
        self.shape_propagator = ShapePropagator(debug=False)
        
        input_shape = model_dict['input']['shape']
        current_shape = (None, *input_shape)
        
        for i, layer in enumerate(model_dict['layers']):
            params = layer.get('params', {})
            if params is None:
                params = {}
            params = params.copy()
            
            # Propagate shape
            current_shape = self.shape_propagator.propagate(
                current_shape, layer, framework='tensorflow'
            )
            
            if layer['type'] == 'Flatten':
                self.layers_list.append(tf.keras.layers.Flatten())
            elif layer['type'] == 'Dense':
                units = params.get('units')
                if units is None:
                    units = trial.suggest_int(f'dense_units_{i}', 64, 256)
                if not isinstance(units, (int, float)):
                    raise InvalidParameterError(
                        parameter='units',
                        value=units,
                        layer_type='Dense',
                        expected='numeric value'
                    )
                units = int(units)
                activation = params.get('activation', 'relu')
                if activation and activation.lower() != 'linear':
                    self.layers_list.append(tf.keras.layers.Dense(units, activation=activation))
                else:
                    self.layers_list.append(tf.keras.layers.Dense(units))
            elif layer['type'] == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    rate = trial.suggest_float(f'dropout_rate_{i}', 0.3, 0.7, step=0.1)
                if not isinstance(rate, (int, float)):
                    raise InvalidParameterError(
                        parameter='rate',
                        value=rate,
                        layer_type='Dropout',
                        expected='numeric value between 0 and 1'
                    )
                self.layers_list.append(tf.keras.layers.Dropout(float(rate)))
            elif layer['type'] == 'Output':
                units = params.get('units', 10)
                if not isinstance(units, (int, float)):
                    raise InvalidParameterError(
                        parameter='units',
                        value=units,
                        layer_type='Output',
                        expected='numeric value'
                    )
                units = int(units)
                activation = params.get('activation', 'softmax')
                if activation and activation.lower() != 'linear':
                    self.layers_list.append(tf.keras.layers.Dense(units, activation=activation))
                else:
                    self.layers_list.append(tf.keras.layers.Dense(units))
            elif layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int(f'conv2d_filters_{i}', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                activation = params.get('activation', 'relu')
                self.layers_list.append(tf.keras.layers.Conv2D(
                    filters, kernel_size, activation=activation if activation else None
                ))
            elif layer['type'] == 'MaxPooling2D':
                pool_size = params.get('pool_size', 2)
                self.layers_list.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
            elif layer['type'] == 'BatchNormalization':
                self.layers_list.append(tf.keras.layers.BatchNormalization())
            elif layer['type'] == 'LSTM':
                units = params.get('units', trial.suggest_int(f'lstm_units_{i}', 32, 256))
                return_sequences = params.get('return_sequences', False)
                self.layers_list.append(
                    tf.keras.layers.LSTM(units, return_sequences=return_sequences)
                )
            else:
                raise ValueError(f"Unsupported layer type for TensorFlow: {layer['type']}")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the model."""
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


# Training Method
def train_model(
    model: Union[DynamicPTModel, DynamicTFModel], 
    optimizer: Union[optim.Optimizer, Any], 
    train_loader: Union[torch.utils.data.DataLoader, tf.data.Dataset], 
    val_loader: Union[torch.utils.data.DataLoader, tf.data.Dataset], 
    backend: str = 'pytorch', 
    epochs: int = 1, 
    execution_config: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, float, float]:
    """
    Train a model and return validation metrics.
    
    Args:
        model: Model to train
        optimizer: Optimizer for training
        train_loader: Training data loader
        val_loader: Validation data loader
        backend: Backend being used ('pytorch' or 'tensorflow')
        epochs: Number of training epochs
        execution_config: Execution configuration (device, etc.)
        
    Returns:
        Tuple of (validation_loss, accuracy, precision, recall)
        
    Raises:
        UnsupportedBackendError: If backend is not supported
    """
    if execution_config is None:
        execution_config = {}
    
    if backend == 'pytorch':
        device = get_device(execution_config.get("device", "auto"))
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        preds, targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())

        preds_np = np.array(preds)
        targets_np = np.array(targets)
        precision = precision_score(targets_np, preds_np, average='macro', zero_division=0)
        recall = recall_score(targets_np, preds_np, average='macro', zero_division=0)

        return val_loss / len(val_loader), correct / total, precision, recall
    
    elif backend == 'tensorflow':
        # TensorFlow training
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(train_loader, epochs=epochs, verbose=0)
        
        # Evaluate
        results = model.evaluate(val_loader, verbose=0)
        val_loss = results[0]
        accuracy = results[1]
        
        # Calculate precision and recall
        preds = []
        targets = []
        for data, target in val_loader:
            output = model(data, training=False)
            pred = tf.argmax(output, axis=1).numpy()
            preds.extend(pred)
            targets.extend(target.numpy())
        
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        precision = precision_score(targets_np, preds_np, average='macro', zero_division=0)
        recall = recall_score(targets_np, preds_np, average='macro', zero_division=0)
        
        return val_loss, accuracy, precision, recall
    else:
        raise UnsupportedBackendError(backend=backend, available_backends=['pytorch', 'tensorflow'])


# HPO Objective
def objective(
    trial: Trial, 
    config: str, 
    dataset_name: str = 'MNIST', 
    backend: str = 'pytorch', 
    device: str = 'auto'
) -> Tuple[float, float, float, float]:
    """
    Objective function for HPO optimization.
    
    Args:
        trial: Optuna trial for suggesting hyperparameters
        config: Neural DSL configuration string
        dataset_name: Name of dataset to use
        backend: Backend to use ('pytorch' or 'tensorflow')
        device: Device to use ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        Tuple of (loss, accuracy, precision, recall) metrics
        
    Raises:
        HPOException: If optimization fails
    """
    try:
        # Parse the network configuration
        model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

        # Resolve batch size from training_config or HPO
        training_config = model_dict.get('training_config', {})
        batch_size = training_config.get('batch_size', 32)

        if isinstance(batch_size, dict) and 'hpo' in batch_size:
            hpo = batch_size['hpo']
            if hpo['type'] == 'categorical':
                batch_size = trial.suggest_categorical("batch_size", hpo['values'])
            elif hpo['type'] == 'range':
                low = hpo.get('start', hpo.get('low', hpo.get('min')))
                high = hpo.get('end', hpo.get('high', hpo.get('max')))
                batch_size = trial.suggest_int("batch_size", low, high, step=hpo.get('step', 1))
            elif hpo['type'] == 'log_range':
                low = hpo.get('start', hpo.get('low', hpo.get('min')))
                high = hpo.get('end', hpo.get('high', hpo.get('max')))
                batch_size = trial.suggest_int("batch_size", low, high, log=True)
        elif isinstance(batch_size, list):
            batch_size = trial.suggest_categorical("batch_size", batch_size)

        batch_size = int(batch_size)

        # Get data loaders
        input_shape = model_dict['input']['shape']
        train_loader = get_data(dataset_name, input_shape, batch_size, True, backend)
        val_loader = get_data(dataset_name, input_shape, batch_size, False, backend)

        # Create the model
        model = create_dynamic_model(model_dict, trial, hpo_params, backend)
        
        # Get optimizer configuration
        optimizer_config = model.model_dict.get('optimizer')
        if optimizer_config is None:
            optimizer_config = {'type': 'Adam', 'params': {'learning_rate': 0.001}}
        elif 'params' not in optimizer_config or not optimizer_config['params']:
            optimizer_config['params'] = {'learning_rate': 0.001}

        lr = optimizer_config['params'].get('learning_rate', 0.001)
        if not isinstance(lr, (int, float)):
            lr = 0.001

        # Create optimizer
        if backend == 'pytorch':
            optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
        elif backend == 'tensorflow':
            optimizer_cls = getattr(tf.keras.optimizers, optimizer_config['type'])
            optimizer = optimizer_cls(learning_rate=lr)
        else:
            raise UnsupportedBackendError(
                backend=backend, available_backends=['pytorch', 'tensorflow']
            )

        # Get device and create execution config
        execution_config = {'device': device}

        # Train the model and get metrics
        loss, acc, precision, recall = train_model(
            model, optimizer, train_loader, val_loader, 
            backend=backend, execution_config=execution_config
        )
        return loss, acc, precision, recall
    
    except Exception as e:
        raise HPOException(f"Objective function failed: {str(e)}") from e


# Optimize and Return
def optimize_and_return(
    config: str, 
    n_trials: int = 10, 
    dataset_name: str = 'MNIST', 
    backend: str = 'pytorch', 
    device: str = 'auto'
) -> Dict[str, Union[int, float]]:
    """
    Run HPO optimization and return the best hyperparameters.
    
    Args:
        config: Neural DSL configuration string
        n_trials: Number of optimization trials to run
        dataset_name: Name of dataset to use
        backend: Backend to use ('pytorch' or 'tensorflow')
        device: Device to use ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        Dictionary of best hyperparameters
        
    Raises:
        HPOException: If optimization fails
    """
    import os
    
    # Set device mode
    if device.lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['NEURAL_FORCE_CPU'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_ENABLE_TENSOR_FLOAT_32_EXECUTION'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    try:
        study = optuna.create_study(directions=["minimize", "maximize", "maximize", "maximize"])

        def objective_wrapper(trial: Trial) -> Tuple[float, float, float, float]:
            # Parse the config once per trial
            model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

            # Resolve batch_size from training_config or HPO
            training_config = model_dict.get('training_config', {})
            batch_size = training_config.get('batch_size', 32)

            if isinstance(batch_size, dict) and 'hpo' in batch_size:
                hpo = batch_size['hpo']
                if hpo['type'] == 'categorical':
                    batch_size = trial.suggest_categorical("batch_size", hpo['values'])
                elif hpo['type'] == 'range':
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    batch_size = trial.suggest_int("batch_size", low, high, step=hpo.get('step', 1))
                elif hpo['type'] == 'log_range':
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    batch_size = trial.suggest_int("batch_size", low, high, log=True)
            elif isinstance(batch_size, list):
                batch_size = trial.suggest_categorical("batch_size", batch_size)

            batch_size = int(batch_size)

            input_shape = model_dict['input']['shape']
            train_loader = get_data(dataset_name, input_shape, batch_size, True, backend)
            val_loader = get_data(dataset_name, input_shape, batch_size, False, backend)

            # Create model and resolve all HPO parameters
            model = create_dynamic_model(model_dict, trial, hpo_params, backend)

            # Get optimizer configuration
            optimizer_config = model.model_dict.get('optimizer')
            if optimizer_config is None:
                optimizer_config = {'type': 'Adam', 'params': {'learning_rate': 0.001}}
            elif 'params' not in optimizer_config or not optimizer_config['params']:
                optimizer_config['params'] = {'learning_rate': 0.001}

            lr = optimizer_config['params'].get('learning_rate', 0.001)
            if not isinstance(lr, (int, float)):
                lr = 0.001

            if backend == 'pytorch':
                optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
            elif backend == 'tensorflow':
                optimizer = getattr(tf.keras.optimizers, optimizer_config['type'])(learning_rate=lr)

            # Train and evaluate
            execution_config = {'device': device}
            loss, acc, precision, recall = train_model(
                model, optimizer, train_loader, val_loader, 
                backend=backend, execution_config=execution_config
            )
            return loss, acc, precision, recall

        study.optimize(objective_wrapper, n_trials=n_trials)

        # Get best trial - we use the first trial in best_trials
        if not study.best_trials:
            raise HPOException("No successful trials completed")
        
        best_trial = study.best_trials[0]
        best_params = best_trial.params

        # Normalize the best parameters
        normalized_params = {
            'batch_size': best_params.get('batch_size', 32),
        }
        
        # Extract layer parameters
        for key, value in best_params.items():
            if key.startswith('Dense_units'):
                normalized_params['dense_units'] = value
            elif key.startswith('Dropout_rate'):
                normalized_params['dropout_rate'] = value
            elif key.startswith('Conv2D_filters'):
                normalized_params['conv2d_filters'] = value
            elif key.startswith('opt_learning_rate'):
                normalized_params['learning_rate'] = value

        # Fallback for learning rate
        if 'learning_rate' not in normalized_params:
            normalized_params['learning_rate'] = 0.001

        return normalized_params
    
    except Exception as e:
        raise HPOException(f"Optimization failed: {str(e)}") from e
