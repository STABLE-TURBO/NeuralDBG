"""Layer processing utilities for Neural DSL parser.

This module contains helper functions for processing and validating layer parameters.
"""

from typing import Dict, List, Any, Tuple, Optional


def extract_ordered_and_named_params(param_values: Any) -> Tuple[List[Any], Dict[str, Any]]:
    """Extract ordered and named parameters from raw parameter values.
    
    Args:
        param_values: Raw parameter values (list, dict, or single value)
        
    Returns:
        Tuple of (ordered_params, named_params)
    """
    ordered_params = []
    named_params = {}
    
    if isinstance(param_values, list):
        for val in param_values:
            if isinstance(val, dict):
                if 'hpo' in val:
                    # HPO expression - needs special handling
                    if len(ordered_params) == 0 and len(named_params) == 0:
                        ordered_params.append(val)
                    else:
                        named_params.update(val)
                else:
                    named_params.update(val)
            elif isinstance(val, list):
                ordered_params.extend(val)
            else:
                ordered_params.append(val)
    elif isinstance(param_values, dict):
        named_params = param_values
    elif param_values is not None:
        ordered_params.append(param_values)
        
    return ordered_params, named_params


def validate_positive_integer(value: Any, param_name: str, layer_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        layer_type: Type of the layer
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Skip validation for HPO parameters
    if isinstance(value, dict) and 'hpo' in value:
        return True, None
        
    if not isinstance(value, int):
        # Handle float that could be integer
        if isinstance(value, float) and value.is_integer():
            return True, None
        return False, f"{layer_type} {param_name} must be an integer, got {value}"
    
    if value <= 0:
        return False, f"{layer_type} {param_name} must be positive, got {value}"
        
    return True, None


def validate_positive_number(value: Any, param_name: str, layer_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        layer_type: Type of the layer
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Skip validation for HPO parameters
    if isinstance(value, dict) and 'hpo' in value:
        return True, None
        
    if not isinstance(value, (int, float)):
        return False, f"{layer_type} {param_name} must be a number, got {value}"
    
    if value <= 0:
        return False, f"{layer_type} {param_name} must be positive, got {value}"
        
    return True, None


def validate_kernel_size(kernel_size: Any, layer_type: str) -> Tuple[bool, Optional[str]]:
    """Validate kernel size parameter.
    
    Args:
        kernel_size: Kernel size value
        layer_type: Type of the layer
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Skip validation for HPO parameters
    if isinstance(kernel_size, dict) and 'hpo' in kernel_size:
        return True, None
        
    if isinstance(kernel_size, (list, tuple)):
        if not all(isinstance(k, int) for k in kernel_size):
            return False, f"{layer_type} kernel_size must be integers, got {kernel_size}"
        if not all(k > 0 for k in kernel_size):
            return False, f"{layer_type} kernel_size should be positive integers, got {kernel_size}"
    elif not isinstance(kernel_size, int) or kernel_size <= 0:
        return False, f"{layer_type} kernel_size must be a positive integer, got {kernel_size}"
        
    return True, None


def validate_pool_size(pool_size: Any, layer_type: str) -> Tuple[bool, Optional[str]]:
    """Validate pool size parameter.
    
    Args:
        pool_size: Pool size value
        layer_type: Type of the layer
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Skip validation for HPO parameters
    if isinstance(pool_size, dict) and 'hpo' in pool_size:
        return True, None
        
    if isinstance(pool_size, (list, tuple)):
        if not all(isinstance(p, int) and p > 0 for p in pool_size):
            return False, f"{layer_type} pool_size must be positive integers, got {pool_size}"
    elif not isinstance(pool_size, int) or pool_size <= 0:
        return False, f"{layer_type} pool_size must be a positive integer, got {pool_size}"
        
    return True, None


def validate_dropout_rate(rate: Any) -> Tuple[bool, Optional[str]]:
    """Validate dropout rate parameter.
    
    Args:
        rate: Dropout rate value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Skip validation for HPO parameters
    if isinstance(rate, dict) and 'hpo' in rate:
        return True, None
        
    if not isinstance(rate, (int, float)):
        return False, f"Dropout rate must be a number, got {rate}"
    
    if not 0 <= rate <= 1:
        return False, f"Dropout rate should be between 0 and 1, got {rate}"
        
    return True, None


def validate_device_specification(device: str) -> Tuple[bool, Optional[str]]:
    """Validate device specification.
    
    Args:
        device: Device specification string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_device_prefixes = ['cpu', 'cuda', 'gpu', 'tpu', 'xla']
    is_valid = any(device.startswith(prefix) for prefix in valid_device_prefixes)
    
    if not is_valid:
        return False, f"Invalid device specification: '{device}'. Valid devices are: {', '.join(valid_device_prefixes)}"
        
    return True, None


def map_positional_to_dense_params(ordered_params: List[Any]) -> Dict[str, Any]:
    """Map positional parameters to Dense layer named parameters.
    
    Args:
        ordered_params: List of positional parameters
        
    Returns:
        Dictionary of named parameters
    """
    params = {}
    if len(ordered_params) >= 1:
        params['units'] = ordered_params[0]
    if len(ordered_params) >= 2:
        params['activation'] = ordered_params[1]
    return params


def map_positional_to_conv2d_params(ordered_params: List[Any]) -> Dict[str, Any]:
    """Map positional parameters to Conv2D layer named parameters.
    
    Args:
        ordered_params: List of positional parameters
        
    Returns:
        Dictionary of named parameters
    """
    params = {}
    if len(ordered_params) >= 1:
        params['filters'] = ordered_params[0]
    if len(ordered_params) >= 2:
        params['kernel_size'] = ordered_params[1]
        if isinstance(params['kernel_size'], (list, tuple)):
            params['kernel_size'] = tuple(params['kernel_size'])
    if len(ordered_params) >= 3:
        params['activation'] = ordered_params[2]
    return params


def map_positional_to_output_params(ordered_params: List[Any]) -> Dict[str, Any]:
    """Map positional parameters to Output layer named parameters.
    
    Args:
        ordered_params: List of positional parameters
        
    Returns:
        Dictionary of named parameters
    """
    params = {}
    if len(ordered_params) >= 1:
        params['units'] = ordered_params[0]
    if len(ordered_params) >= 2:
        params['activation'] = ordered_params[1]
    return params


def map_positional_to_multiheadattention_params(ordered_params: List[Any]) -> Dict[str, Any]:
    """Map positional parameters to MultiHeadAttention layer named parameters.
    
    Args:
        ordered_params: List of positional parameters
        
    Returns:
        Dictionary of named parameters
    """
    params = {}
    if len(ordered_params) >= 1:
        params['num_heads'] = ordered_params[0]
    if len(ordered_params) >= 2:
        params['key_dim'] = ordered_params[1]
    if len(ordered_params) >= 3:
        params['value_dim'] = ordered_params[2]
    return params


def extract_device_from_items(items: List[Any], extract_value_fn) -> Optional[str]:
    """Extract device specification from parse tree items.
    
    Args:
        items: List of parse tree items
        extract_value_fn: Function to extract values from nodes
        
    Returns:
        Device specification string or None
    """
    if len(items) > 1:
        device_node = items[-1]
        return extract_value_fn(device_node) if device_node else None
    return None


def merge_param_dicts(param_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of parameter dictionaries.
    
    Args:
        param_list: List of parameter dictionaries
        
    Returns:
        Merged dictionary
    """
    merged = {}
    for param_dict in param_list:
        if isinstance(param_dict, dict):
            merged.update(param_dict)
    return merged


def normalize_1d_shape(shape: Any) -> Any:
    """Normalize 1D shape tuples to single values.
    
    Args:
        shape: Shape tuple or value
        
    Returns:
        Normalized shape (single value if 1D tuple, otherwise unchanged)
    """
    if isinstance(shape, tuple) and len(shape) == 1:
        return shape[0]
    return shape
