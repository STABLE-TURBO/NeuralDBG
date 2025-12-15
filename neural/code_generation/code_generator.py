import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neural.exceptions import (
    CodeGenException,
    FileOperationError,
    InvalidParameterError,
    UnsupportedBackendError,
    UnsupportedLayerError,
)
from neural.parser.parser import ModelTransformer, create_parser
from neural.shape_propagation.shape_propagator import ShapePropagator


# Set up logging
logger = logging.getLogger(__name__)
# Set the logger level to WARNING to reduce debug output
logger.setLevel(logging.WARNING)

# Supported layer types for code generation
SUPPORTED_LAYERS = {
    'Embedding',
    'PositionalEncoding',
    'GlobalAveragePooling1D',
    'TransformerEncoder',
    'TransformerDecoder',
    'BatchNormalization',
    'Conv2D',
    'Dense',
    'MaxPooling2D',
    'AveragePooling2D',
    'Flatten',
    'LSTM',
    'GRU',
    'Dropout',
    'Output',
    'Residual',
    'Add',
    'Concatenate',
}

def to_number(x: str) -> Union[int, float]:
    try:
        return int(x)
    except ValueError:
        return float(x)


# --- Policy helpers (extracted for clarity and testing) ---

def _policy_ensure_2d_before_dense_tf(
    rank_non_batch: int,
    auto_flatten_output: bool,
    propagator: ShapePropagator,
    current_input_shape: Tuple[Optional[int], ...],
) -> Tuple[str, Tuple[Optional[int], ...]]:
    """Ensure 2D input for Dense/Output in TF.

    Returns (insert_code, updated_shape). insert_code is a string to append to the TF code.
    Raises CodeGenException when higher-rank input is not allowed and auto_flatten_output is False.
    """
    if rank_non_batch > 1:
        if auto_flatten_output:
            # Insert Flatten layer and propagate shape safely
            insert = "x = layers.Flatten()(x)\n"
            try:
                current_input_shape = propagator.propagate(current_input_shape, {"type": "Flatten"}, framework='tensorflow')
            except Exception as e:
                logger.warning(f"Shape propagation warning (auto-flatten): {e}")
            return insert, current_input_shape
        raise CodeGenException(
            "Layer 'Output' expects 2D input (batch, features) but got higher-rank. "
            "Insert a Flatten/GAP before it or pass auto_flatten_output=True.",
            backend='tensorflow',
            layer_type='Output'
        )
    return "", current_input_shape


def _policy_ensure_2d_before_dense_pt(
    rank_non_batch: int,
    auto_flatten_output: bool,
    forward_code_body: List[str],
    propagator: ShapePropagator,
    current_input_shape: Tuple[Optional[int], ...],
) -> Tuple[Optional[int], ...]:
    """Ensure 2D input for Dense/Output in PyTorch.

    Mutates forward_code_body when flatten is inserted and returns the updated shape.
    Raises CodeGenException when higher-rank input is not allowed and auto_flatten_output is False.
    """
    if rank_non_batch > 1:
        if auto_flatten_output:
            forward_code_body.append("x = x.view(x.size(0), -1)  # Flatten input")
            try:
                current_input_shape = propagator.propagate(current_input_shape, {"type": "Flatten"}, framework='pytorch')
            except Exception as e:
                logger.warning(f"Shape propagation warning (auto-flatten): {e}")
            return current_input_shape
        raise CodeGenException(
            "Layer 'Output' expects 2D input (batch, features) but got higher-rank. "
            "Insert a Flatten/GAP before it or pass auto_flatten_output=True.",
            backend='pytorch',
            layer_type='Output'
        )
    return current_input_shape

def generate_code(model_data: Dict[str, Any], backend: str, best_params: Optional[Dict[str, Any]] = None, auto_flatten_output: bool = False) -> str:
    if not isinstance(model_data, dict) or 'layers' not in model_data or 'input' not in model_data:
        raise InvalidParameterError(
            parameter='model_data',
            value=model_data,
            expected="dict with 'layers' and 'input' keys"
        )

    # Check if auto_flatten_output is specified in model_data
    if 'auto_flatten_output' in model_data:
        auto_flatten_output = model_data['auto_flatten_output']

    indent = "    "
    propagator = ShapePropagator(debug=False)
    # Initial input shape includes batch dimension: (None, channels, height, width)
    input_shape_tuple = tuple(model_data['input']['shape'])
    if input_shape_tuple and input_shape_tuple[0] is None:
        current_input_shape = input_shape_tuple
    else:
        current_input_shape = (None,) + input_shape_tuple

    # Process expanded layers before modifying model_dat
    # Expand layers based on 'multiply' key
    expanded_layers = []
    for layer in model_data.get('layers', []):
        # Validate layer format with default to dictionary
        if not isinstance(layer, dict) or 'type' not in layer:
            raise InvalidParameterError(
                parameter='layer',
                value=layer,
                expected="dict with 'type' key"
            )
        # Default Multiply Value to 1
        multiply = layer.get('multiply', 1)
        if not isinstance(multiply, int) or multiply < 1:
            raise InvalidParameterError(
                parameter='multiply',
                value=multiply,
                layer_type=layer.get('type'),
                expected="positive integer"
            )
        # Shallow copy to avoid modifying original layer
        layer_copy = layer.copy()
        # Changes to nested mutable objects will affect both copies
        if 'multiply' in layer_copy:
            del layer_copy['multiply']
        for _ in range(multiply):
            expanded_layers.append(layer_copy.copy())

    # Store original layers
    original_layers = model_data['layers']
    model_data['layers'] = expanded_layers

    if backend == "tensorflow":
        optimizer_config = model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config

        code = "import tensorflow as tf\nfrom tensorflow.keras import layers\n"
        code += f"from tensorflow.keras.optimizers import {optimizer_type}\n"
        code += "from neural.tracking.experiment_tracker import ExperimentManager\n\n"

        code += "# Initialize Experiment Tracking\n"
        code += "experiment_manager = ExperimentManager()\n"
        code += "experiment = experiment_manager.create_experiment()\n"
        code += "experiment.log_hyperparameters({'optimizer': '" + optimizer_type + "', 'backend': 'tensorflow'})\n\n"

        code += "# Custom Callback for Tracking\n"
        code += "class NeuralTrackingCallback(tf.keras.callbacks.Callback):\n"
        code += "    def __init__(self, experiment):\n"
        code += "        super().__init__()\n"
        code += "        self.experiment = experiment\n\n"
        code += "    def on_epoch_end(self, epoch, logs=None):\n"
        code += "        if logs:\n"
        code += "            self.experiment.log_metrics(logs, step=epoch)\n\n"

        # Add input shape handling
        input_shape = tuple(model_data['input']['shape'])  # Use model-defined shape (no batch dim here)
        code += f"# Input layer with shape {input_shape}\n"
        code += f"inputs = layers.Input(shape={input_shape})\n"
        code += "x = inputs\n\n"

        for layer in expanded_layers:
            layer_type = layer['type']
            params = layer.get('params', {})

            # Emit warning for unsupported layer types
            if layer_type not in SUPPORTED_LAYERS:
                warnings.warn(
                    f"Layer type '{layer_type}' is not officially supported and may not generate correct code.",
                    UserWarning,
                    stacklevel=2
                )

            # Policy: Dense/Output require 2D input (batch, features). Use helper for TF policy.
            try:
                rank_non_batch = max(0, len(current_input_shape) - 1)
            except Exception:
                rank_non_batch = 0
            if backend == "tensorflow" and layer_type in ("Dense", "Output"):
                insert_code, current_input_shape = _policy_ensure_2d_before_dense_tf(
                    rank_non_batch, auto_flatten_output, propagator, current_input_shape
                )
                code += insert_code

            if layer_type == "Residual":
                code += "# Residual block\n"
                code += "residual_input = x\n"
                for sub_layer in layer.get('sub_layers', []):
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {})
                    layer_code = generate_tensorflow_layer(sub_type, sub_params, best_params)
                    if layer_code:
                        if ('\n' in layer_code) or ('x =' in layer_code):
                            code += layer_code + "\n"
                        else:
                            code += f"x = {layer_code}(x)\n"
                code += "x = layers.Add()([x, residual_input])\n"
            else:
                layer_code = generate_tensorflow_layer(layer_type, params, best_params)
                if layer_code:
                    if ('\n' in layer_code) or ('x =' in layer_code):
                        code += layer_code + "\n"
                    else:
                        code += f"x = {layer_code}(x)\n"
            try:
                current_input_shape = propagator.propagate(current_input_shape, layer, framework='tensorflow')
            except Exception as e:
                logger.warning(f"Shape propagation warning: {e}")

        code += "\n# Build model\n"
        code += "model = tf.keras.Model(inputs=inputs, outputs=x)\n"

        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {}).items():
                # Handle HPO parameters
                if isinstance(v, dict):
                    if 'hpo' in v and best_params and k in best_params:
                        v = best_params[k]
                    elif 'value' in v:
                        v = v['value']
                    else:
                        continue
                opt_params.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
        loss_entry = model_data.get('loss', {'value': 'categorical_crossentropy'})
        if loss_entry is None or not isinstance(loss_entry, (str, dict)):
            loss_value = 'categorical_crossentropy'  # Fallback
        elif isinstance(loss_entry, str):
            loss_value = loss_entry
        else:
            loss_value = loss_entry.get('value', 'categorical_crossentropy')
        code += f"# Compile model with {optimizer_type} optimizer and {loss_value} loss\n"
        code += f"model.compile(loss='{loss_value}', optimizer={optimizer_type}({', '.join(opt_params)}))\n"

        if 'training_config' in model_data:
            tc = model_data['training_config']
            code += "# Training configuration\n"
            code += (
                f"model.fit(\n    x_train, y_train,\n"
                f"    epochs={tc.get('epochs', 10)},\n"
                f"    batch_size={tc.get('batch_size', 32)},\n"
                f"    validation_split={tc.get('validation_split', 0.2)},\n"
                f"    callbacks=[NeuralTrackingCallback(experiment)],\n"
                f"    verbose=1\n)\n"
            )
            if 'training_config' in model_data and model_data['training_config'].get('mixed_precision', False):
                code = "from tensorflow.keras.mixed_precision import set_global_policy\n" + code
                code += "set_global_policy('mixed_float16')\n"
            if 'training_config' in model_data and 'save_path' in model_data['training_config']:
                code += f"model.save('{model_data['training_config']['save_path']}')\n"
        return code

    elif backend == "pytorch":
        optimizer_config = model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config
        code = "import logging\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torchvision.transforms as transforms\nimport math\n"
        code += "from torchvision import datasets\n"
        code += "from torch.utils.data import DataLoader\n"
        code += "from neural.tracking.experiment_tracker import ExperimentManager\n\n"
        code += "logger = logging.getLogger(__name__)\n\n"

        code += "# Initialize Experiment Tracking\n"
        code += "experiment_manager = ExperimentManager()\n"
        code += "experiment = experiment_manager.create_experiment()\n"
        code += "experiment.log_hyperparameters({'optimizer': '" + optimizer_type + "', 'backend': 'pytorch'})\n\n"
        
        # Check if we need positional encoding classes
        needs_positional_encoding = any(layer.get('type') == 'PositionalEncoding' for layer in expanded_layers)
        if needs_positional_encoding:
            code += "# Sinusoidal Positional Encoding\n"
            code += "class SinusoidalPositionalEncoding(nn.Module):\n"
            code += "    def __init__(self, max_len=5000):\n"
            code += "        super(SinusoidalPositionalEncoding, self).__init__()\n"
            code += "        self.max_len = max_len\n\n"
            code += "    def forward(self, x):\n"
            code += "        batch_size, seq_len, d_model = x.size()\n"
            code += "        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)\n"
            code += "        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / d_model))\n"
            code += "        pos_encoding = torch.zeros(seq_len, d_model, device=x.device)\n"
            code += "        pos_encoding[:, 0::2] = torch.sin(position * div_term)\n"
            code += "        pos_encoding[:, 1::2] = torch.cos(position * div_term)\n"
            code += "        return x + pos_encoding.unsqueeze(0)\n\n"
            code += "# Learnable Positional Encoding\n"
            code += "class LearnablePositionalEncoding(nn.Module):\n"
            code += "    def __init__(self, max_len=5000, d_model=512):\n"
            code += "        super(LearnablePositionalEncoding, self).__init__()\n"
            code += "        self.max_len = max_len\n"
            code += "        self.d_model = d_model\n"
            code += "        self.pos_embedding = None\n\n"
            code += "    def forward(self, x):\n"
            code += "        batch_size, seq_len, d_model = x.size()\n"
            code += "        if self.pos_embedding is None or self.pos_embedding.size(0) != self.max_len or self.pos_embedding.size(1) != d_model:\n"
            code += "            self.pos_embedding = nn.Parameter(torch.randn(self.max_len, d_model, device=x.device))\n"
            code += "        positions = self.pos_embedding[:seq_len, :].unsqueeze(0)\n"
            code += "        return x + positions\n\n"
        
        code += "# Neural network model definition\n"
        code += "class NeuralNetworkModel(nn.Module):\n"
        code += f"{indent}def __init__(self):\n"
        code += f"{indent}{indent}super(NeuralNetworkModel, self).__init__()\n"

        layers_code = []
        forward_code_body: list[str] = []
        layer_counts = {}

        for i, layer in enumerate(expanded_layers):
            layer_type = layer['type']
            params = layer.get('params', {})
            if params is not None:
                params = params.copy()
            else:
                params = {}

            # Emit warning for unsupported layer types
            if layer_type not in SUPPORTED_LAYERS:
                warnings.warn(
                    f"Layer type '{layer_type}' is not officially supported and may not generate correct code.",
                    UserWarning,
                    stacklevel=2
                )

            # Policy: Dense/Output require 2D input (batch, features). Use helper for PT policy.
            try:
                rank_non_batch = max(0, len(current_input_shape) - 1)
            except Exception:
                rank_non_batch = 0
            if layer_type in ("Dense", "Output"):
                current_input_shape = _policy_ensure_2d_before_dense_pt(
                    rank_non_batch, auto_flatten_output, forward_code_body, propagator, current_input_shape
                )

            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0

            layer_name = f"layer{i}_{layer_type.lower()}"
            layer_counts[layer_type] += 1

            if layer_type == "Residual":
                residual_layers = []
                residual_forward = []
                for sub_layer in layer.get('sub_layers', []):
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {})
                    sub_layer_code = generate_pytorch_layer(sub_type, sub_params, current_input_shape)
                    if sub_layer_code:
                        residual_layers.append(sub_layer_code)
                    try:
                        current_input_shape = propagator.propagate(current_input_shape, sub_layer, framework='pytorch')
                    except Exception as e:
                        logger.warning(f"Shape propagation warning in residual: {e}")
                if residual_layers:
                    layers_code.append(f"self.{layer_name} = nn.Sequential({', '.join(residual_layers)})")
                    forward_code_body.append(f"x = x + self.{layer_name}(x)")
            elif layer_type == "Dense":
                # If first layer or previous layer requires flattening
                if i == 0 or expanded_layers[i-1]['type'] in ["Input", "Flatten"]:
                    # product of current_input_shape elements over specified axis
                    # Ensure all dimensions are integers before calculating product
                    dims = []
                    logger.warning(f"Current input shape: {current_input_shape}")
                    for dim in current_input_shape[1:]:
                        if dim is not None:
                            # Handle dictionary values
                            if isinstance(dim, dict):
                                # If it's a dictionary with a 'value' key, use that value
                                if 'value' in dim:
                                    dims.append(dim['value'])
                                # Otherwise, use a default value
                                else:
                                    logger.warning(f"Dictionary dimension without 'value' key: {dim}, using default")
                                    dims.append(64)  # Default value
                            elif isinstance(dim, (int, float)):
                                dims.append(dim)
                            else:
                                logger.warning(f"Unexpected dimension type: {type(dim)}, value: {dim}, using default")
                                dims.append(64)  # Default value
                        else:
                            logger.warning("None dimension found, skipping")
                    logger.warning(f"Dimensions after processing: {dims}")
                    in_features = np.prod(dims) if dims else 64
                    # Modified to evict None values and NoneType * Int Errors
                    # The first input shape tuple in the list for tensorflow contains None
                else:
                    # Get the previous layer's output features
                    in_features = current_input_shape[-1]
                    # Handle dictionary values
                    if isinstance(in_features, dict):
                        # If it's a dictionary with a 'value' key, use that value
                        if 'value' in in_features:
                            in_features = in_features['value']
                        # Otherwise, use a default value
                        else:
                            logger.warning(f"Dictionary dimension without 'value' key: {in_features}, using default")
                            in_features = 64  # Default value
                out_features = params.get("units", 64)
                # Handle dictionary values in out_features
                if isinstance(out_features, dict):
                    # If it's a dictionary with a 'value' key, use that value
                    if 'value' in out_features:
                        out_features = out_features['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                        out_features = 64
                layer_code = f"nn.Linear(in_features={in_features}, out_features={out_features})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "Dropout":
                rate = params.get("rate", 0.5)
                # Handle dictionary values in rate
                if isinstance(rate, dict):
                    # If it's a dictionary with a 'value' key, use that value
                    if 'value' in rate:
                        rate = rate['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {rate}, using default")
                        rate = 0.5
                layer_code = f"nn.Dropout(p={rate})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "Output":
                # Get the previous layer's output features
                in_features = current_input_shape[-1]
                # Handle dictionary values
                if isinstance(in_features, dict):
                    # If it's a dictionary with a 'value' key, use that value
                    if 'value' in in_features:
                        in_features = in_features['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary dimension without 'value' key: {in_features}, using default")
                        in_features = 64  # Default value
                out_features = params.get("units", 10)
                # Handle dictionary values in out_features
                if isinstance(out_features, dict):
                    # If it's a dictionary with a 'value' key, use that value
                    if 'value' in out_features:
                        out_features = out_features['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                        out_features = 10
                activation = params.get("activation", "softmax")
                # Handle dictionary values in activation
                if isinstance(activation, dict):
                    # If it's a dictionary with a 'value' key, use that value
                    if 'value' in activation:
                        activation = activation['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {activation}, using default")
                        activation = "softmax"
                if activation == "softmax":
                    layer_code = f"nn.Sequential(nn.Linear(in_features={in_features}, out_features={out_features}), nn.Softmax(dim=1))"
                else:
                    layer_code = f"nn.Linear(in_features={in_features}, out_features={out_features})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            else:
                # Use generate_pytorch_layer for other layer types
                layer_code = generate_pytorch_layer(layer_type, params, current_input_shape, best_params)
                if layer_code:
                    if layer_type == "MultiHeadAttention":
                        layers_code.append(f"self.{layer_name} = {layer_code}")
                        mode = params.get("mode", "self")
                        if mode == "cross":
                            forward_code_body.append(f"x, _ = self.{layer_name}(x, context, context)")
                        else:
                            forward_code_body.append(f"x, _ = self.{layer_name}(x, x, x)")
                    elif layer_type in ("TransformerEncoder", "TransformerDecoder", "Embedding", "PositionalEncoding"):
                        layers_code.append(f"self.{layer_name} = {layer_code}")
                        forward_code_body.append(f"x = self.{layer_name}(x)")
                    else:
                        layers_code.append(f"self.{layer_name} = {layer_code}")
                        forward_code_body.append(f"x = self.{layer_name}(x)")

            try:
                current_input_shape = propagator.propagate(current_input_shape, layer, framework='pytorch')
            except Exception as e:
                logger.warning(f"Shape propagation warning: {e}")

        model_data['layers'] = original_layers

        for line in layers_code:
            code += f"{indent}{indent}{line}\n"
        code += f"\n{indent}# Forward pass\n"
        code += f"{indent}def forward(self, x):\n"
        if expanded_layers and expanded_layers[0]['type'] == 'Dense':
            code += f"{indent}{indent}x = x.view(x.size(0), -1)  # Flatten input\n"
        for line in forward_code_body:
            code += f"{indent}{indent}{line}\n"
        code += f"{indent}{indent}return x\n\n"

        code += "# Model instantiation\n"
        code += "model = NeuralNetworkModel()\n"
        code += "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        code += "model.to(device)\n\n"

        code += "# MNIST dataset\n"
        code += "transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n"
        code += "train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n"
        batch_size = model_data.get('training_config', {}).get('batch_size', 64)
        if best_params and 'batch_size' in best_params:
            batch_size = best_params['batch_size']
        code += f"train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)\n\n"

        loss_entry = model_data.get('loss', {'value': 'crossentropy'})
        if loss_entry is None or not isinstance(loss_entry, (str, dict)):
            loss_value = 'crossentropy'
        elif isinstance(loss_entry, str):
            loss_value = loss_entry
        else:
            loss_value = loss_entry.get('value', 'crossentropy')
        loss_fn = "nn.CrossEntropyLoss()" if "crossentropy" in loss_value.lower() else "nn.MSELoss()"
        code += f"# Loss function\nloss_fn = {loss_fn}\n"

        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {'lr': 0.001}).items():
                param_name = 'lr' if k == 'learning_rate' else k
                # Handle dictionary values in optimizer parameters
                if isinstance(v, dict):
                    # If it's a dictionary with HPO, use the best parameter if available
                    if 'hpo' in v and best_params and 'learning_rate' in best_params:
                        v = best_params['learning_rate']
                    # If it's a dictionary with a 'value' key, use that value
                    elif 'value' in v:
                        v = v['value']
                    # Otherwise, use a default value
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {v}, using default")
                        v = 0.001  # Default learning rate
                opt_params.append(f"{param_name}={repr(v)}")
        code += f"# Optimizer\noptimizer = optim.{optimizer_type}(model.parameters(), {', '.join(opt_params)})\n"

        if 'training_config' in model_data:
            tc = model_data['training_config']
            code += "\n# Mixed precision training setup\n"
            code += "scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')\n"
            code += f"for epoch in range({tc.get('epochs', 10)}):\n"
            code += f"{indent}running_loss = 0.0\n"  # Add loss tracking
            code += f"{indent}for batch_idx, (data, target) in enumerate(train_loader):\n"
            code += f"{indent}{indent}data, target = data.to(device), target.to(device)\n"
            code += f"{indent}{indent}optimizer.zero_grad()\n"
            code += f"{indent}{indent}with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):\n"
            code += f"{indent}{indent}{indent}output = model(data)\n"
            code += f"{indent}{indent}{indent}loss = loss_fn(output, target)\n"
            code += f"{indent}{indent}scaler.scale(loss).backward()\n"
            code += f"{indent}{indent}scaler.step(optimizer)\n"
            code += f"{indent}{indent}scaler.update()\n"
            code += f"{indent}{indent}running_loss += loss.item()  # Accumulate loss\n"
            code += f"{indent}avg_loss = running_loss / len(train_loader)\n"
            code += f"{indent}print(f'Epoch {{epoch+1}}/{{{tc.get('epochs', 10)}}} - Loss: {{avg_loss:.4f}}')\n"  # Print average loss
            code += "\n# Evaluate model\n"
            code += "model.eval()\n"
            code += "correct = 0\n"
            code += "total = 0\n"
            code += "with torch.no_grad():\n"
            code += f"{indent}for data, target in train_loader:\n"
            code += f"{indent}{indent}data, target = data.to(device), target.to(device)\n"
            code += f"{indent}{indent}outputs = model(data)\n"
            code += f"{indent}{indent}_, predicted = torch.max(outputs.data, 1)\n"
            code += f"{indent}{indent}total += target.size(0)\n"
            code += f"{indent}{indent}correct += (predicted == target).sum().item()\n"
            code += f"{indent}accuracy = 100 * correct / total\n"
            code += "print(f'Accuracy: {accuracy:.2f}%')\n"
            code += "experiment.log_metrics({'loss': avg_loss, 'accuracy': accuracy}, step=epoch)\n"
            if 'save_path' in tc:
                code += f"{indent}{indent}torch.save(model.state_dict(), '{tc['save_path']}')\n"


        return code

    elif backend == "onnx":
        return export_onnx(model_data, "model.onnx")

    else:
        raise UnsupportedBackendError(
            backend=backend,
            available_backends=['tensorflow', 'pytorch', 'onnx']
        )

def save_file(filename: str, content: str) -> None:
    """Save content to a file."""
    import os
    try:
        # Create parent directories if they don't exist
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(content)
    except Exception as e:
        raise FileOperationError(
            operation='write',
            filepath=filename,
            reason=str(e)
        )
    logger.info(f"Successfully saved file: {filename}")

def load_file(filename: str) -> Any:
    """Load and parse a neural config file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        raise FileOperationError(
            operation='read',
            filepath=filename,
            reason=str(e)
        )
    if filename.endswith('.neural') or filename.endswith('.nr'):
        return create_parser('network').parse(content)
    elif filename.endswith('.rnr'):
        return create_parser('research').parse(content)
    else:
        raise FileOperationError(
            operation='parse',
            filepath=filename,
            reason="Unsupported file type. Expected .neural, .nr, or .rnr"
        )

def generate_onnx(model_data: Dict[str, Any]) -> Any:
    """Generate ONNX model"""
    # Import ONNX only when needed (avoid hard dependency for TF/PyTorch paths)
    from onnx import TensorProto, helper
    # Create nodes for each layer
    nodes = []
    current_input = "input"
    output_shape = list(model_data["input"]["shape"])  # Track output shape

    for i, layer in enumerate(model_data['layers']):
        layer_type = layer['type']
        params = layer.get('params', {})
        output_name = f"layer_{i}_output"

        if layer_type == "Conv2D":
            nodes.append(helper.make_node(
                'Conv',
                inputs=[current_input],
                outputs=[output_name],
                kernel_shape=params.get('kernel_size', [3, 3]),
                strides=params.get('strides', [1, 1])
            ))
            # Update output shape for Conv2D (keeps spatial dims, changes channels)
            if len(output_shape) == 4:
                output_shape[-1] = params.get('filters', 32)
        elif layer_type == "Output":
            units = params.get('units', 10)
            nodes.append(helper.make_node(
                'Gemm',
                inputs=[current_input],
                outputs=[output_name]
            ))
            # Output layer produces (batch, units)
            output_shape = [output_shape[0], units]
        # Add other layer types as needed

        current_input = output_name

    # Create graph with nodes
    graph = helper.make_graph(
        nodes=nodes,
        name="NeuralModel",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, model_data["input"]["shape"])],
        outputs=[helper.make_tensor_value_info(current_input, TensorProto.FLOAT, output_shape)],
        initializer=[]
    )

    # Create model
    model = helper.make_model(graph, producer_name="Neural")
    model.opset_import[0].version = 13

    return model

def export_onnx(model_data: Dict[str, Any], filename: str = "model.onnx") -> str:
    """Export model to ONNX format."""
    import onnx
    model = generate_onnx(model_data)
    onnx.save(model, filename)
    return f"ONNX model saved to {filename}"

def _extract_param_value(param_value, default_value, best_params=None, param_name=None):
    """Extract the actual value from a parameter that might be a dict with HPO info."""
    if isinstance(param_value, dict):
        # Check if this is an HPO parameter and we have best_params
        if 'hpo' in param_value and best_params and param_name and param_name in best_params:
            return best_params[param_name]
        # Otherwise use the value field
        elif 'value' in param_value:
            return param_value['value']
        else:
            logger.warning(f"Dictionary parameter without 'value' key: {param_value}, using default")
            return default_value
    return param_value

def generate_tensorflow_layer(layer_type, params, best_params=None):
    """Generate TensorFlow layer code"""
    if layer_type == "Embedding":
        input_dim = _extract_param_value(params.get("input_dim", 10000), 10000, best_params, "input_dim")
        output_dim = _extract_param_value(params.get("output_dim", 128), 128, best_params, "output_dim")
        mask_zero = _extract_param_value(params.get("mask_zero", False), False, best_params, "mask_zero")
        input_length = _extract_param_value(params.get("input_length", None), None, best_params, "input_length")
        
        code = f"layers.Embedding(input_dim={input_dim}, output_dim={output_dim}"
        if mask_zero:
            code += f", mask_zero={mask_zero}"
        if input_length:
            code += f", input_length={input_length}"
        code += ")"
        return code
    elif layer_type == "MultiHeadAttention":
        num_heads = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        key_dim = _extract_param_value(params.get("key_dim", 64), 64, best_params, "key_dim")
        value_dim = _extract_param_value(params.get("value_dim", None), None, best_params, "value_dim")
        dropout = _extract_param_value(params.get("dropout", 0.0), 0.0, best_params, "dropout")
        use_bias = _extract_param_value(params.get("use_bias", True), True, best_params, "use_bias")
        mode = _extract_param_value(params.get("mode", "self"), "self", best_params, "mode")
        
        value_dim_str = f", value_dim={value_dim}" if value_dim else ""
        dropout_str = f", dropout={dropout}" if dropout > 0 else ""
        use_bias_str = f", use_bias={use_bias}" if not use_bias else ""
        
        if mode == "cross":
            return f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim}{value_dim_str}{dropout_str}{use_bias_str})(x, context)"
        else:
            return f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim}{value_dim_str}{dropout_str}{use_bias_str})(x, x)"
    elif layer_type == "TransformerEncoder":
        num_heads = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        ff_dim = _extract_param_value(params.get("ff_dim", 512), 512, best_params, "ff_dim")
        dropout = _extract_param_value(params.get("dropout", 0.1), 0.1, best_params, "dropout")
        num_layers = _extract_param_value(params.get("num_layers", 1), 1, best_params, "num_layers")
        activation = _extract_param_value(params.get("activation", "relu"), "relu", best_params, "activation")
        use_attention_mask = _extract_param_value(params.get("use_attention_mask", False), False, best_params, "use_attention_mask")
        
        code = ["# TransformerEncoder block"]
        
        if use_attention_mask:
            code.append("# Attention mask should be provided as input")
            code.append("attention_mask = None  # Set this to your mask tensor")
        
        for layer_idx in range(num_layers):
            code.append(f"# Encoder Layer {layer_idx + 1}")
            code.append("x = layers.LayerNormalization(epsilon=1e-6)(x)")
            
            if use_attention_mask:
                code.append(f"attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={ff_dim})(x, x, attention_mask=attention_mask)")
            else:
                code.append(f"attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={ff_dim})(x, x)")
            
            code.append(f"attn_output = layers.Dropout({dropout})(attn_output)")
            code.append("x = layers.Add()([x, attn_output])")
            code.append("x = layers.LayerNormalization(epsilon=1e-6)(x)")
            code.append(f"ffn_output = layers.Dense({ff_dim}, activation='{activation}')(x)")
            code.append(f"ffn_output = layers.Dense({ff_dim})(ffn_output)")
            code.append(f"ffn_output = layers.Dropout({dropout})(ffn_output)")
            code.append("x = layers.Add()([x, ffn_output])")
        
        return "\n".join(code)
    elif layer_type == "TransformerDecoder":
        num_heads = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        ff_dim = _extract_param_value(params.get("ff_dim", 512), 512, best_params, "ff_dim")
        dropout = _extract_param_value(params.get("dropout", 0.1), 0.1, best_params, "dropout")
        d_model = _extract_param_value(params.get("d_model", ff_dim), ff_dim, best_params, "d_model")
        use_causal_mask = _extract_param_value(params.get("use_causal_mask", True), True, best_params, "use_causal_mask")
        
        code = [
            "# TransformerDecoder block with cross-attention",
            "# Self-attention with causal masking",
            "decoder_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)",
        ]
        if use_causal_mask:
            code.append("# Apply causal mask for autoregressive decoding")
            code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}, use_causal_mask=True)(decoder_norm1, decoder_norm1)")
        else:
            code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm1, decoder_norm1)")
        code.extend([
            f"x = layers.Add()([x, layers.Dropout({dropout})(self_attn_output)])",
            "# Cross-attention with encoder output (assume encoder_output available)",
            "decoder_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)",
            f"cross_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm2, encoder_output, encoder_output)",
            f"x = layers.Add()([x, layers.Dropout({dropout})(cross_attn_output)])",
            "# Feed-forward network",
            "decoder_norm3 = layers.LayerNormalization(epsilon=1e-6)(x)",
            f"ff_output = layers.Dense({ff_dim}, activation='relu')(decoder_norm3)",
            f"ff_output = layers.Dense({d_model})(ff_output)",
            f"x = layers.Add()([x, layers.Dropout({dropout})(ff_output)])"
        ])
        return "\n".join(code)
    elif layer_type == "PositionalEncoding":
        max_len = _extract_param_value(params.get("max_len", 5000), 5000, best_params, "max_len")
        encoding_type = _extract_param_value(params.get("encoding_type", "sinusoidal"), "sinusoidal", best_params, "encoding_type")
        
        if encoding_type == "sinusoidal":
            code = [
                "# Sinusoidal Positional Encoding",
                "import numpy as np",
                f"def get_positional_encoding(seq_len, d_model, max_len={max_len}):",
                "    position = np.arange(seq_len)[:, np.newaxis]",
                "    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))",
                "    pos_encoding = np.zeros((seq_len, d_model))",
                "    pos_encoding[:, 0::2] = np.sin(position * div_term)",
                "    pos_encoding[:, 1::2] = np.cos(position * div_term)",
                "    return tf.constant(pos_encoding, dtype=tf.float32)",
                "seq_len = tf.shape(x)[1]",
                "d_model = tf.shape(x)[2]",
                "pos_encoding = get_positional_encoding(seq_len, d_model)",
                "x = x + pos_encoding"
            ]
            return "\n".join(code)
        else:
            code = [
                "# Learnable Positional Encoding",
                f"pos_embedding = layers.Embedding(input_dim={max_len}, output_dim=tf.shape(x)[2])",
                "seq_len = tf.shape(x)[1]",
                "positions = tf.range(start=0, limit=seq_len, delta=1)",
                "x = x + pos_embedding(positions)"
            ]
            return "\n".join(code)
    elif layer_type == "BatchNormalization":
        momentum = _extract_param_value(params.get("momentum", 0.99), 0.99, best_params, "momentum")
        epsilon = _extract_param_value(params.get("epsilon", 0.001), 0.001, best_params, "epsilon")
        if momentum == 0.99 and epsilon == 0.001:
            return "layers.BatchNormalization()"
        return f"layers.BatchNormalization(momentum={momentum}, epsilon={epsilon})"
    elif layer_type == "Conv2D":
        filters = _extract_param_value(params.get("filters", 32), 32, best_params, "filters")
        kernel_size = _extract_param_value(params.get("kernel_size", (3, 3)), (3, 3), best_params, "kernel_size")
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]
        padding = _extract_param_value(params.get("padding", "same"), "same", best_params, "padding")
        activation = _extract_param_value(params.get("activation", None), None, best_params, "activation")
        code = f"layers.Conv2D(filters={filters}, kernel_size={kernel_size}, padding='{padding}'"
        if activation:
            code += f", activation='{activation}'"
        code += ")"
        return code
    elif layer_type == "Dense":
        units = _extract_param_value(params.get("units", 64), 64, best_params, "units")
        activation = _extract_param_value(params.get("activation", None), None, best_params, "activation")
        code = f"layers.Dense(units={units}"
        if activation:
            code += f", activation='{activation}'"
        code += ")"
        return code
    elif layer_type == "MaxPooling2D":
        pool_size = _extract_param_value(params.get("pool_size", (2, 2)), (2, 2), best_params, "pool_size")
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size
        strides = _extract_param_value(params.get("strides", None), None, best_params, "strides")
        if strides:
            return f"layers.MaxPooling2D(pool_size={pool_size}, strides={strides})"
        return f"layers.MaxPooling2D(pool_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = _extract_param_value(params.get("pool_size", (2, 2)), (2, 2), best_params, "pool_size")
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size[0] if isinstance(pool_size[0], int) else pool_size
        return f"layers.AveragePooling2D(pool_size={pool_size})"
    elif layer_type == "Flatten":
        return "layers.Flatten()"
    elif layer_type == "LSTM":
        units = _extract_param_value(params.get("units", 128), 128, best_params, "units")
        return_sequences = _extract_param_value(params.get("return_sequences", False), False, best_params, "return_sequences")
        return f"layers.LSTM(units={units}, return_sequences={str(return_sequences)})"
    elif layer_type == "GRU":
        units = _extract_param_value(params.get("units", 64), 64, best_params, "units")
        return_sequences = _extract_param_value(params.get("return_sequences", False), False, best_params, "return_sequences")
        return f"layers.GRU(units={units}, return_sequences={str(return_sequences)})"
    elif layer_type == "Dropout":
        rate = _extract_param_value(params.get("rate", 0.5), 0.5, best_params, "rate")
        return f"layers.Dropout(rate={rate})"
    elif layer_type == "GlobalAveragePooling1D":
        return "layers.GlobalAveragePooling1D()"
    elif layer_type == "GlobalAveragePooling2D":
        return "layers.GlobalAveragePooling2D()"
    elif layer_type == "GlobalAveragePooling3D":
        return "layers.GlobalAveragePooling3D()"
    elif layer_type == "GlobalMaxPooling1D":
        return "layers.GlobalMaxPooling1D()"
    elif layer_type == "GlobalMaxPooling2D":
        return "layers.GlobalMaxPooling2D()"
    elif layer_type == "GlobalMaxPooling3D":
        return "layers.GlobalMaxPooling3D()"
    elif layer_type == "LayerNormalization":
        epsilon = _extract_param_value(params.get("epsilon", 0.001), 0.001, best_params, "epsilon")
        return f"layers.LayerNormalization(epsilon={epsilon})"
    elif layer_type == "Output":
        units = _extract_param_value(params.get("units", 10), 10, best_params, "units")
        activation = _extract_param_value(params.get("activation", "softmax"), "softmax", best_params, "activation")
        return f"layers.Dense(units={units}, activation='{activation}')"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for tensorflow. Skipping.", UserWarning)
        return None


# Pytorch Layers Code Generator
def generate_pytorch_layer(layer_type, params, input_shape: Optional[tuple] = None, best_params=None):
    """Generate PyTorch layer code"""
    if layer_type == "Embedding":
        num_embeddings = _extract_param_value(params.get("input_dim", 1000), 1000, best_params, "input_dim")
        embedding_dim = _extract_param_value(params.get("output_dim", 128), 128, best_params, "output_dim")
        return f"nn.Embedding(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})"
    elif layer_type == "MultiHeadAttention":
        embed_dim = _extract_param_value(params.get("embed_dim", None), None, best_params, "embed_dim")
        num_heads = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        dropout = _extract_param_value(params.get("dropout", 0.0), 0.0, best_params, "dropout")
        batch_first = _extract_param_value(params.get("batch_first", True), True, best_params, "batch_first")
        
        if embed_dim is None and input_shape is not None and len(input_shape) >= 2:
            embed_dim = input_shape[-1]
            if isinstance(embed_dim, dict):
                embed_dim = _extract_param_value(embed_dim, 512, best_params, "embed_dim")
        elif embed_dim is None:
            embed_dim = 512
        
        return f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}, batch_first={batch_first})"
    elif layer_type == "TransformerEncoder":
        d_model = _extract_param_value(params.get("d_model", 512), 512, best_params, "d_model")
        nhead = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        dim_feedforward = _extract_param_value(params.get("ff_dim", 2048), 2048, best_params, "ff_dim")
        dropout = _extract_param_value(params.get("dropout", 0.1), 0.1, best_params, "dropout")
        num_layers = _extract_param_value(params.get("num_layers", 1), 1, best_params, "num_layers")
        activation = _extract_param_value(params.get("activation", "relu"), "relu", best_params, "activation")
        
        if num_layers > 1:
            return f"nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}'), num_layers={num_layers})"
        else:
            return f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}')"
    elif layer_type == "TransformerDecoder":
        d_model = _extract_param_value(params.get("d_model", 512), 512, best_params, "d_model")
        nhead = _extract_param_value(params.get("num_heads", 8), 8, best_params, "num_heads")
        dim_feedforward = _extract_param_value(params.get("ff_dim", 2048), 2048, best_params, "ff_dim")
        dropout = _extract_param_value(params.get("dropout", 0.1), 0.1, best_params, "dropout")
        return f"nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout})"
    elif layer_type == "PositionalEncoding":
        max_len = _extract_param_value(params.get("max_len", 5000), 5000, best_params, "max_len")
        encoding_type = _extract_param_value(params.get("encoding_type", "sinusoidal"), "sinusoidal", best_params, "encoding_type")
        
        if encoding_type == "sinusoidal":
            return f"SinusoidalPositionalEncoding(max_len={max_len})"
        else:
            return f"LearnablePositionalEncoding(max_len={max_len})"
    elif layer_type == "Conv2D":
        data_format = _extract_param_value(params.get("data_format", "channels_last"), "channels_last", best_params, "data_format")
        in_channels = 3  # Default value
        if input_shape is not None:
            in_channels = input_shape[1] if data_format == "channels_first" else input_shape[3]
            in_channels = in_channels if len(input_shape) > 3 else 3
        out_channels = _extract_param_value(params.get("filters", 32), 32, best_params, "filters")
        kernel_size = _extract_param_value(params.get("kernel_size", 3), 3, best_params, "kernel_size")
        # Handle both tuple/list and integer kernel sizes
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]  # Use first element for both dimensions
        return f"nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size})"
    elif layer_type == "BatchNormalization":
        data_format = _extract_param_value(params.get("data_format", "channels_last"), "channels_last", best_params, "data_format")
        if input_shape and len(input_shape) > 3:
            num_features = input_shape[1] if data_format == "channels_first" else input_shape[3]
        else:
            # Use the number of filters from previous Conv2D layer if available
            num_features = _extract_param_value(params.get("filters", 64), 64, best_params, "filters")
        momentum = _extract_param_value(params.get("momentum", 0.9), 0.9, best_params, "momentum")
        eps = _extract_param_value(params.get("epsilon", 0.001), 0.001, best_params, "epsilon")
        # Only include momentum and eps if they differ from defaults
        if momentum == 0.9 and eps == 0.001:
            return f"nn.BatchNorm2d(num_features={num_features})"
        return f"nn.BatchNorm2d(num_features={num_features}, momentum={momentum}, eps={eps})"
    elif layer_type == "Dense":
        # Calculate in_features with proper handling of dictionary values
        if input_shape:
            # Ensure all dimensions are integers before calculating product
            dims = []
            for dim in input_shape[1:]:
                if dim is not None:
                    dims.append(_extract_param_value(dim, 64, best_params, "dim"))
            in_features = np.prod(dims) if dims else 64
        else:
            in_features = 64
        out_features = _extract_param_value(params.get("units", 64), 64, best_params, "units")
        activation = _extract_param_value(params.get("activation", None), None, best_params, "activation")
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation:
            if activation == "relu":
                layers.append("nn.ReLU()")
            elif activation == "tanh":
                layers.append("nn.Tanh()")
            elif activation == "softmax":
                layers.append("nn.Softmax(dim=1)")
            elif activation == "invalid":
                layers.append("nn.Identity()")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "MaxPooling2D":
        pool_size = _extract_param_value(params.get("pool_size", 2), 2, best_params, "pool_size")
        # Handle both tuple/list and integer pool sizes
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        strides = _extract_param_value(params.get("strides", None), None, best_params, "strides")
        if strides:
            return f"nn.MaxPool2d(kernel_size={pool_size}, stride={strides})"
        return f"nn.MaxPool2d(kernel_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = _extract_param_value(params.get("pool_size", 2), 2, best_params, "pool_size")
        # Handle both tuple/list and integer pool sizes
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        return f"nn.AvgPool2d(kernel_size={pool_size})"
    elif layer_type == "Flatten":
        return "nn.Flatten()"
    elif layer_type == "Dropout":
        rate = _extract_param_value(params.get("rate", 0.5), 0.5, best_params, "rate")
        return f"nn.Dropout(p={rate})"
    elif layer_type == "Output":
        # Calculate in_features with proper handling of dictionary values
        if input_shape:
            # Ensure all dimensions are integers before calculating product
            dims = []
            for dim in input_shape[1:]:
                if dim is not None:
                    dims.append(_extract_param_value(dim, 64, best_params, "dim"))
            in_features = np.prod(dims) if dims else 64
        else:
            in_features = 64
        out_features = _extract_param_value(params.get("units", 10), 10, best_params, "units")
        activation = _extract_param_value(params.get("activation", "softmax"), "softmax", best_params, "activation")
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation == "softmax":
            layers.append("nn.Softmax(dim=1)")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "LSTM":
        # Get input size with proper handling of dictionary values
        if input_shape:
            input_size = input_shape[-1]
            input_size = _extract_param_value(input_size, 32, best_params, "input_size")
        else:
            input_size = 32

        hidden_size = _extract_param_value(params.get("units", 128), 128, best_params, "units")
        return f"nn.LSTM(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "GRU":
        input_size = _extract_param_value(params.get("input_size", 128), 128, best_params, "input_size")
        hidden_size = _extract_param_value(params.get("units", 64), 64, best_params, "units")
        return f"nn.GRU(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "GlobalAveragePooling1D":
        return "nn.AdaptiveAvgPool1d(1)"
    elif layer_type == "GlobalAveragePooling2D":
        return "nn.AdaptiveAvgPool2d(1)"
    elif layer_type == "GlobalAveragePooling3D":
        return "nn.AdaptiveAvgPool3d(1)"
    elif layer_type == "GlobalMaxPooling1D":
        return "nn.AdaptiveMaxPool1d(1)"
    elif layer_type == "GlobalMaxPooling2D":
        return "nn.AdaptiveMaxPool2d(1)"
    elif layer_type == "GlobalMaxPooling3D":
        return "nn.AdaptiveMaxPool3d(1)"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for pytorch. Skipping.", UserWarning)
        return None

## Optimized Code Generation ##

def generate_optimized_dsl(config: str, best_params: Dict[str, Any]) -> str:
    """Generate optimized DSL code with the best hyperparameters."""
    try:
        transformer = ModelTransformer()
        _, hpo_params = transformer.parse_network_with_hpo(config)
        lines = config.strip().split('\n')

        logger.info(f"Initial lines: {lines}")
        logger.info(f"best_params: {best_params}")
        logger.info(f"hpo_params: {hpo_params}")

        # Process all HPO parameters uniformly
        for hpo in hpo_params:
            # Determine param_key based on layer_type
            if hpo['layer_type'].lower() == 'training_config' and hpo['param_name'] == 'batch_size':
                param_key = 'batch_size'
            elif hpo['layer_type'].lower() == 'optimizer' and hpo['param_name'] == 'params.learning_rate':
                param_key = 'learning_rate'
            else:
                param_key = f"{hpo['layer_type'].lower()}_{hpo['param_name']}"

            if param_key not in best_params:
                logger.warning(f"Parameter {param_key} not found in best_params, skipping")
                continue

            if 'hpo' not in hpo or not hpo['hpo']:
                logger.warning(f"Missing 'hpo' data for parameter {param_key}, skipping")
                continue

            # Construct the HPO string based on type
            hpo_type = hpo['hpo'].get('type')
            if not hpo_type:
                logger.warning(f"Missing 'type' in hpo data for parameter {param_key}, skipping")
                continue

            if hpo_type in ('choice', 'categorical'):
                values = hpo['hpo'].get('original_values', hpo['hpo'].get('values', []))
                if not values:
                    logger.warning(f"Missing 'values' for choice/categorical parameter {param_key}, skipping")
                    continue
                hpo_str = f"choice({', '.join(map(str, values))})"
            elif hpo_type == 'range':
                start = hpo['hpo'].get('start')
                end = hpo['hpo'].get('end')
                original_parts = hpo['hpo'].get('original_parts', [])
                if not original_parts and (start is None or end is None):
                    logger.warning(f"Missing range bounds for parameter {param_key}, skipping")
                    continue
                if not original_parts:
                    original_parts = [str(start), str(end)]
                if 'step' in hpo['hpo']:
                    hpo_str = f"range({', '.join(original_parts)}, step={hpo['hpo']['step']})"
                else:
                    hpo_str = f"range({', '.join(original_parts)})"
            elif hpo_type == 'log_range':
                # Try both naming conventions (start/end and min/max) for backward compatibility
                low = hpo['hpo'].get('original_low', str(hpo['hpo'].get('start', hpo['hpo'].get('min', ''))))
                high = hpo['hpo'].get('original_high', str(hpo['hpo'].get('end', hpo['hpo'].get('max', ''))))
                if not low or not high:
                    logger.warning(f"Missing log_range bounds for parameter {param_key}, skipping")
                    continue
                hpo_str = f"log_range({low}, {high})"
            else:
                logger.warning(f"Unknown HPO type: {hpo_type}, skipping")
                continue

            # Replace the entire HPO expression
            logger.info(f"Processing hpo: {hpo}, param_key: {param_key}, hpo_str: {hpo_str}")
            for i, line in enumerate(lines):
                full_hpo = f"HPO({hpo_str})"
                if full_hpo in line:
                    old_line = lines[i]
                    # Ensure the parameter value is properly converted to a string
                    param_value = best_params[param_key]
                    # Handle different types of parameter values
                    if isinstance(param_value, (int, float)):
                        param_value_str = str(param_value)
                    elif isinstance(param_value, str):
                        param_value_str = f'"{param_value}"'  # Add quotes for string values
                    elif isinstance(param_value, dict):
                        # Convert dictionary to a string representation
                        param_value_str = str(_extract_param_value(param_value, param_value, best_params, param_key))
                    else:
                        param_value_str = str(param_value)

                    new_line = line.replace(full_hpo, param_value_str)
                    lines[i] = new_line
                    logger.info(f"Replaced line {i}: '{old_line}' -> '{new_line}'")
                    break

        # Special case for optimizer learning rate
        if 'learning_rate' in best_params:
            for i, line in enumerate(lines):
                if 'optimizer:' in line and 'learning_rate=HPO(' in line:
                    old_line = lines[i]
                    # Create a completely new optimizer line with the correct syntax
                    optimizer_type = re.search(r'optimizer:\s*(\w+)\(', old_line)
                    if optimizer_type:
                        opt_type = optimizer_type.group(1)
                        # Handle different types of learning rate values
                        lr_value = best_params['learning_rate']
                        if isinstance(lr_value, (int, float)):
                            lr_str = str(lr_value)
                        elif isinstance(lr_value, str):
                            lr_str = f'"{lr_value}"'  # Add quotes for string values
                        elif isinstance(lr_value, dict):
                            lr_str = str(_extract_param_value(lr_value, lr_value, best_params, "learning_rate"))
                        else:
                            lr_str = str(lr_value)

                        new_line = f"        optimizer: {opt_type}(learning_rate={lr_str})"
                        lines[i] = new_line
                        logger.info(f"Replaced optimizer line {i}: '{old_line}' -> '{new_line}'")
                        break

        logger.info(f"Final lines: {lines}")
        return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Error generating optimized DSL: {str(e)}")
        raise
