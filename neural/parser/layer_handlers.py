"""Layer-specific handlers for Neural DSL parser.

This module contains specialized processing functions for complex layer types
like Conv2D, Dense, LSTM, etc.
"""

from typing import Dict, Any, List, Optional, Tuple
from . import layer_processors as lp
from . import hpo_utils


def process_dense_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process Dense layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
        
    Raises:
        Validation error if required parameters are missing
    """
    ordered_params, named_params = lp.extract_ordered_and_named_params(param_values)
    
    # Handle list of HPO expressions
    if isinstance(param_values, list):
        for val in param_values:
            if isinstance(val, dict) and 'hpo' in val:
                if 'units' not in named_params:
                    named_params['units'] = val
                else:
                    raise_error_fn("Multiple HPO expressions not supported as positional args", node)
    
    # Map positional parameters
    params = lp.map_positional_to_dense_params(ordered_params)
    
    # Check max positional params
    is_valid, error_msg = lp.validate_param_count(ordered_params, 2, "Dense")
    if not is_valid:
        raise_error_fn(error_msg, node)
    
    # Merge named parameters
    params.update(named_params)
    
    # Validate units parameter
    if 'units' not in params:
        raise_error_fn("Dense layer requires 'units' parameter", node)
    
    units = params['units']
    if isinstance(units, dict) and 'hpo' in units:
        track_hpo_fn('Dense', 'units', units, node)
    elif isinstance(units, list) and len(units) > 1:
        # Handle list as categorical HPO
        if all(isinstance(u, (int, float)) for u in units):
            hpo_config = hpo_utils.create_categorical_hpo_from_list(units)
            params['units'] = hpo_config
            track_hpo_fn('Dense', 'units', hpo_config, node)
    else:
        # Validate units is a number
        if isinstance(units, str):
            raise_error_fn("Dense units must be a number", node)
        is_valid, error_msg = lp.validate_positive_integer(units, 'units', 'Dense')
        if not is_valid:
            raise_error_fn(error_msg, node)
        params['units'] = int(units) if isinstance(units, float) and units.is_integer() else units
    
    return params


def process_conv2d_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process Conv2D layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
        
    Raises:
        Validation error if required parameters are missing or invalid
    """
    # Check for padding HPO in nested list
    if isinstance(param_values, list) and len(param_values) > 0:
        if isinstance(param_values[0], list):
            for param in param_values[0]:
                if isinstance(param, dict) and 'padding' in param:
                    if isinstance(param['padding'], dict) and 'hpo' in param['padding']:
                        track_hpo_fn('Conv2D', 'padding', param['padding'], node)
    
    # Extract ordered and named parameters
    ordered_params = []
    named_params = {}
    
    if isinstance(param_values, list):
        for param in param_values:
            if isinstance(param, dict):
                named_params.update(param)
            else:
                ordered_params.append(param)
    elif isinstance(param_values, dict):
        named_params = param_values
    else:
        ordered_params.append(param_values)
    
    # Map positional parameters
    params = lp.map_positional_to_conv2d_params(ordered_params)
    params.update(named_params)
    
    # Validate filters parameter
    if 'filters' not in params:
        raise_error_fn("Conv2D layer requires 'filters' parameter", node)
    
    filters = params['filters']
    if isinstance(filters, dict) and 'hpo' in filters:
        track_hpo_fn('Conv2D', 'filters', filters, node)
    elif isinstance(filters, list):
        # Handle list with HPO
        hpo_config = hpo_utils.extract_hpo_from_list(filters)
        if hpo_config:
            track_hpo_fn('Conv2D', 'filters', hpo_config, node)
            params['filters'] = hpo_config
        elif len(filters) > 0 and isinstance(filters[0], (int, float)):
            params['filters'] = filters[0]
            if filters[0] <= 0:
                raise_error_fn(f"Conv2D filters must be a positive integer, got {filters[0]}", node)
        else:
            raise_error_fn(f"Conv2D filters must be a positive integer, got {filters}", node)
    else:
        is_valid, error_msg = lp.validate_positive_integer(filters, 'filters', 'Conv2D')
        if not is_valid:
            raise_error_fn(error_msg, node)
    
    # Validate kernel_size if present
    if 'kernel_size' in params:
        ks = params['kernel_size']
        
        # Handle HPO
        if isinstance(ks, dict) and 'hpo' in ks:
            track_hpo_fn('Conv2D', 'kernel_size', ks, node)
        elif isinstance(ks, (list, tuple)) and len(ks) > 0:
            if isinstance(ks[0], dict) and 'hpo' in ks[0]:
                track_hpo_fn('Conv2D', 'kernel_size', ks[0], node)
                params['kernel_size'] = ks[0]
            else:
                # Validate as tuple
                is_valid, error_msg = lp.validate_kernel_size(ks, 'Conv2D')
                if not is_valid:
                    raise_error_fn(error_msg, node)
                params['kernel_size'] = tuple(ks)
        else:
            is_valid, error_msg = lp.validate_kernel_size(ks, 'Conv2D')
            if not is_valid:
                raise_error_fn(error_msg, node)
    
    # Track all HPO parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('Conv2D', param_name, param_value, node)
        # Check nested HPO
        elif isinstance(param_value, dict):
            for nested_name, nested_value in param_value.items():
                if isinstance(nested_value, dict) and 'hpo' in nested_value:
                    track_hpo_fn('Conv2D', f"{param_name}.{nested_name}", nested_value, node)
    
    return params


def process_dropout_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process Dropout layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
    """
    params = {}
    
    # Handle list parameters
    if isinstance(param_values, list):
        merged_params = {}
        for elem in param_values:
            if isinstance(elem, dict):
                if 'hpo' in elem:
                    merged_params['rate'] = elem
                else:
                    merged_params.update(elem)
            else:
                merged_params['rate'] = elem
        param_values = merged_params
    
    # Process based on type
    if isinstance(param_values, dict):
        params = param_values.copy()
        if 'rate' in params:
            if isinstance(params['rate'], dict) and 'hpo' in params['rate']:
                track_hpo_fn('Dropout', 'rate', params['rate'], node)
            else:
                is_valid, error_msg = lp.validate_dropout_rate(params['rate'])
                if not is_valid:
                    raise_error_fn(error_msg, node)
        else:
            raise_error_fn("Dropout requires a 'rate' parameter", node)
    elif isinstance(param_values, (int, float)):
        params['rate'] = param_values
        is_valid, error_msg = lp.validate_dropout_rate(params['rate'])
        if not is_valid:
            raise_error_fn(error_msg, node)
    else:
        raise_error_fn("Invalid parameters for Dropout", node)
    
    # Track any remaining HPO parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('Dropout', param_name, param_value, node)
    
    return params


def process_lstm_params(param_values: Any, raise_error_fn, node) -> Dict[str, Any]:
    """Process LSTM layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
    """
    params = {}
    
    if isinstance(param_values, list):
        for val in param_values:
            if isinstance(val, dict):
                params.update(val)
            else:
                if 'units' not in params:
                    params['units'] = val
    elif isinstance(param_values, dict):
        params = param_values
    else:
        params['units'] = param_values
    
    # Validate units
    if 'units' not in params:
        raise_error_fn("LSTM requires 'units' parameter", node)
    
    units = params['units']
    # Skip validation for HPO
    if not (isinstance(units, dict) and 'hpo' in units):
        is_valid, error_msg = lp.validate_positive_integer(units, 'units', 'LSTM')
        if not is_valid:
            raise_error_fn(error_msg, node)
        params['units'] = int(units)
    
    return params


def process_output_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process Output layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
    """
    ordered_params = []
    named_params = {}
    
    if isinstance(param_values, list):
        for val in param_values:
            if isinstance(val, dict):
                if 'hpo' in val and len(named_params) == 0 and len(ordered_params) == 0:
                    named_params['units'] = val
                else:
                    named_params.update(val)
            else:
                ordered_params.append(val)
    elif isinstance(param_values, dict):
        named_params = param_values
    
    # Map positional parameters
    params = lp.map_positional_to_output_params(ordered_params)
    
    # Check max positional params
    if len(ordered_params) > 2:
        raise_error_fn("Output layer accepts at most two positional arguments (units, activation)", node)
    
    params.update(named_params)
    
    # Validate units
    if 'units' not in params:
        raise_error_fn("Output layer requires 'units' parameter", node)
    
    # Track HPO if present
    if isinstance(params['units'], dict) and 'hpo' in params['units']:
        track_hpo_fn('Output', 'units', params['units'], node)
    
    return params


def process_maxpooling2d_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process MaxPooling2D layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
    """
    params = {}
    
    # Extract parameter values
    if hasattr(param_values, 'children'):
        param_vals = [val for val in param_values.children]
    else:
        param_vals = [param_values] if not isinstance(param_values, list) else param_values
    
    # Check if all are dictionaries (named params)
    if all(isinstance(p, dict) for p in param_vals):
        params = lp.merge_param_dicts(param_vals)
    else:
        # Map positional parameters
        if len(param_vals) >= 1:
            params["pool_size"] = param_vals[0]
        if len(param_vals) >= 2:
            params["strides"] = param_vals[1]
        if len(param_vals) >= 3:
            params["padding"] = param_vals[2]
    
    # Validate pool_size
    if "pool_size" in params:
        pool_size = params["pool_size"]
        if isinstance(pool_size, dict) and 'hpo' in pool_size:
            track_hpo_fn('MaxPooling2D', 'pool_size', pool_size, node)
        else:
            is_valid, error_msg = lp.validate_pool_size(pool_size, 'MaxPooling2D')
            if not is_valid:
                raise_error_fn(error_msg, node)
    else:
        raise_error_fn("Missing required parameter 'pool_size'", node)
    
    # Track all HPO parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('MaxPooling2D', param_name, param_value, node)
    
    return params


def process_positionalencoding_params(param_values: Any, raise_error_fn, track_hpo_fn, node) -> Dict[str, Any]:
    """Process PositionalEncoding layer parameters.
    
    Args:
        param_values: Raw parameter values from parsing
        raise_error_fn: Function to raise validation errors
        track_hpo_fn: Function to track HPO parameters
        node: Parse tree node for error reporting
        
    Returns:
        Dictionary of processed parameters
    """
    ordered_params = []
    named_params = {}
    
    if isinstance(param_values, list):
        for val in param_values:
            if isinstance(val, dict):
                if 'hpo' in val and len(named_params) == 0 and len(ordered_params) == 0:
                    named_params['max_len'] = val
                else:
                    named_params.update(val)
            else:
                ordered_params.append(val)
    elif isinstance(param_values, dict):
        named_params = param_values
    elif param_values is not None:
        ordered_params.append(param_values)
    
    # Map positional parameters: max_len, encoding_type
    params = {}
    if len(ordered_params) >= 1:
        params['max_len'] = ordered_params[0]
    if len(ordered_params) >= 2:
        params['encoding_type'] = ordered_params[1]
    
    if len(ordered_params) > 2:
        raise_error_fn("PositionalEncoding layer accepts at most two positional arguments (max_len, encoding_type)", node)
    
    params.update(named_params)
    
    # Set defaults
    if 'max_len' not in params:
        params['max_len'] = 5000
    if 'encoding_type' not in params:
        params['encoding_type'] = 'sinusoidal'
    
    # Validate max_len
    max_len = params['max_len']
    if isinstance(max_len, dict) and 'hpo' in max_len:
        track_hpo_fn('PositionalEncoding', 'max_len', max_len, node)
    elif not isinstance(max_len, (int, float)) or max_len <= 0:
        raise_error_fn(f"PositionalEncoding max_len must be a positive integer, got {max_len}", node)
    
    # Validate encoding_type
    encoding_type = params['encoding_type']
    if isinstance(encoding_type, dict) and 'hpo' in encoding_type:
        track_hpo_fn('PositionalEncoding', 'encoding_type', encoding_type, node)
    elif encoding_type not in ['sinusoidal', 'learnable']:
        raise_error_fn(f"PositionalEncoding encoding_type must be 'sinusoidal' or 'learnable', got {encoding_type}", node)
    
    # Track all HPO parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('PositionalEncoding', param_name, param_value, node)
    
    return params
