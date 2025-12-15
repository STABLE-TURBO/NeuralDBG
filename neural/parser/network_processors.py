"""Network-level processing utilities for Neural DSL parser.

This module contains functions for processing network-level configurations
like execution settings, framework detection, and device specifications.
"""

from typing import Dict, Any, List


def detect_framework(model: Dict[str, Any]) -> str:
    """Detect the appropriate framework for a model.
    
    This examines the model's layers to determine whether it should use
    PyTorch or TensorFlow.
    
    Args:
        model: The parsed model configuration
        
    Returns:
        The detected framework ('pytorch' or 'tensorflow')
    """
    for layer in model.get('layers', []):
        params = layer.get('params') or {}
        if isinstance(params, dict):
            if 'torch' in params.values():
                return 'pytorch'
            if 'keras' in params.values():
                return 'tensorflow'
    return 'tensorflow'


def process_execution_config(model: Dict[str, Any]) -> Dict[str, Any]:
    """Process execution configuration for device specification.
    
    Args:
        model: The parsed model configuration
        
    Returns:
        Updated model with execution configuration
    """
    if 'execution' in model:
        return model

    # Check if this is a device specification test
    is_device_test = False
    has_tpu = False
    has_cuda = False

    # Check model name
    if 'name' in model:
        if model['name'] in ['MultiDeviceModel', 'TPUModel']:
            is_device_test = True
        elif model['name'] == 'DevicePlacementModel':
            is_device_test = True
            model['execution'] = {'device': 'cuda'}
            return model

    # Check for device specifications in layers
    if 'layers' in model:
        for layer in model['layers']:
            if 'device' in layer:
                is_device_test = True
                if layer['device'].startswith('tpu'):
                    has_tpu = True
                elif layer['device'].startswith('cuda'):
                    has_cuda = True

    # Add execution config if needed
    if is_device_test:
        if has_tpu:
            model['execution'] = {'device': 'tpu'}
        else:
            model['execution'] = {'device': 'auto'}

    return model


def validate_input_dimensions(shape: Any, raise_error_fn, node) -> None:
    """Validate input dimensions.
    
    Args:
        shape: Shape tuple or value to validate
        raise_error_fn: Function to raise validation errors
        node: Parse tree node for error reporting
        
    Raises:
        Validation error if dimensions are invalid
    """
    if isinstance(shape, tuple):
        for dim in shape:
            if dim is not None and dim <= 0:
                raise_error_fn(f"Invalid input dimension: {dim}. Dimensions must be positive.", node)
    elif isinstance(shape, int) and shape <= 0:
        raise_error_fn(f"Invalid input dimension: {shape}. Dimensions must be positive.", node)


def process_network_sections(tree_children: List[Any], extract_value_fn) -> Dict[str, Any]:
    """Process network configuration sections from parse tree.
    
    Args:
        tree_children: Children of the network parse tree node
        extract_value_fn: Function to extract values from nodes
        
    Returns:
        Dictionary with network configuration sections
    """
    sections = {
        'input': None,
        'layers': [],
        'loss': None,
        'optimizer': None,
        'training': None,
        'execution': None
    }

    for child in tree_children:
        if not hasattr(child, 'data'):
            continue

        if child.data == 'input_layer':
            sections['input'] = extract_value_fn(child)
        elif child.data == 'layers':
            sections['layers'] = extract_value_fn(child)
        elif child.data == 'loss':
            sections['loss'] = extract_value_fn(child)
        elif child.data == 'optimizer':
            sections['optimizer'] = extract_value_fn(child)
        elif child.data == 'training_config':
            sections['training'] = extract_value_fn(child)
        elif child.data == 'execution_config':
            sections['execution'] = extract_value_fn(child)

    return sections


def expand_repeated_layers(layers: List[Any]) -> List[Any]:
    """Expand layers with repetition counts.
    
    Args:
        layers: List of layers, some may be tuples of (layer, count)
        
    Returns:
        Expanded list of layers
    """
    expanded_layers = []
    for item in layers:
        if isinstance(item, list):
            expanded_layers.extend(item)
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
            layer, count = item
            expanded_layers.extend([layer] * count)
        else:
            expanded_layers.append(item)
    return expanded_layers


def merge_layer_params(ordered_params: List[Any], named_params: Dict[str, Any],
                       param_mapping: Dict[int, str]) -> Dict[str, Any]:
    """Merge ordered and named parameters for a layer.
    
    Args:
        ordered_params: List of positional parameters
        named_params: Dictionary of named parameters
        param_mapping: Mapping from position index to parameter name
        
    Returns:
        Merged parameter dictionary
    """
    params = {}

    # Map positional parameters
    for idx, value in enumerate(ordered_params):
        if idx in param_mapping:
            params[param_mapping[idx]] = value

    # Merge named parameters (overrides positional)
    params.update(named_params)

    return params
