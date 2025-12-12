"""HPO network processing for Neural DSL parser.

This module contains functions for processing networks with HPO parameters,
including optimizer HPO tracking and learning rate schedule processing.
"""

from typing import Dict, Any, Callable
from . import hpo_utils


def process_optimizer_hpo(optimizer_info: Any, track_hpo_fn: Callable) -> None:
    """Process HPO parameters in optimizer configuration.
    
    Args:
        optimizer_info: Optimizer configuration (string or dict)
        track_hpo_fn: Function to track HPO parameters
    """
    # Handle string-based optimizer with HPO expressions
    if isinstance(optimizer_info, str):
        hpo_utils.track_hpo_in_optimizer_string(optimizer_info, track_hpo_fn)
    
    # Handle dictionary-based optimizer
    elif isinstance(optimizer_info, dict) and 'params' in optimizer_info:
        params = optimizer_info['params']
        hpo_utils.track_hpo_in_optimizer_params(params, track_hpo_fn)
        
        # Process learning rate schedule strings
        if 'learning_rate' in params and isinstance(params['learning_rate'], str):
            lr_value = params['learning_rate']
            if '(' in lr_value and ')' in lr_value and 'HPO(' in lr_value:
                hpo_utils.track_hpo_in_lr_schedule_string(lr_value, track_hpo_fn)


def process_training_hpo(training_config: Dict[str, Any], track_hpo_fn: Callable) -> None:
    """Process HPO parameters in training configuration.
    
    Args:
        training_config: Training configuration dictionary
        track_hpo_fn: Function to track HPO parameters
    """
    if not isinstance(training_config, dict):
        return
    
    for param_name, param_value in training_config.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('training', param_name, param_value, None)
        # Handle list of HPO expressions
        elif isinstance(param_value, list):
            for idx, item in enumerate(param_value):
                if isinstance(item, dict) and 'hpo' in item:
                    track_hpo_fn('training', f'{param_name}[{idx}]', item, None)


def process_loss_hpo(loss_config: Any, track_hpo_fn: Callable) -> None:
    """Process HPO parameters in loss configuration.
    
    Args:
        loss_config: Loss configuration (string, dict, or list)
        track_hpo_fn: Function to track HPO parameters
    """
    if isinstance(loss_config, dict) and 'hpo' in loss_config:
        track_hpo_fn('loss', 'function', loss_config, None)
    elif isinstance(loss_config, list):
        for item in loss_config:
            if isinstance(item, dict) and 'hpo' in item:
                track_hpo_fn('loss', 'function', item, None)


def collect_layer_hpo_params(layers: list, existing_hpo_params: list) -> list:
    """Collect all HPO parameters from layers.
    
    Args:
        layers: List of layer configurations
        existing_hpo_params: Existing HPO parameters list
        
    Returns:
        Combined list of HPO parameters
    """
    layer_hpo = []
    
    for layer in layers:
        if not isinstance(layer, dict):
            continue
            
        layer_type = layer.get('type', 'Unknown')
        params = layer.get('params')
        
        if not isinstance(params, dict):
            continue
        
        # Check each parameter for HPO
        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                hpo_entry = {
                    'layer_type': layer_type,
                    'param_name': param_name,
                    'path': f"{layer_type}.{param_name}",
                    'hpo': param_value['hpo'],
                    'node': None
                }
                # Check for duplicates
                if not any(
                    e['layer_type'] == hpo_entry['layer_type'] and
                    e['param_name'] == hpo_entry['param_name'] and
                    str(e['hpo']) == str(hpo_entry['hpo'])
                    for e in existing_hpo_params
                ):
                    layer_hpo.append(hpo_entry)
    
    return layer_hpo


def build_hpo_search_space(hpo_params: list) -> Dict[str, Any]:
    """Build HPO search space from collected parameters.
    
    Args:
        hpo_params: List of HPO parameter configurations
        
    Returns:
        Dictionary representing the search space
    """
    search_space = {}
    
    for param in hpo_params:
        path = param['path']
        hpo_config = param['hpo']
        
        if hpo_config['type'] == 'range':
            search_space[path] = {
                'type': 'uniform',
                'low': hpo_config['start'],
                'high': hpo_config['end'],
                'step': hpo_config.get('step')
            }
        elif hpo_config['type'] == 'log_range':
            search_space[path] = {
                'type': 'loguniform',
                'low': hpo_config['min'],
                'high': hpo_config['max']
            }
        elif hpo_config['type'] == 'categorical':
            search_space[path] = {
                'type': 'categorical',
                'choices': hpo_config['values']
            }
        else:
            # Unknown HPO type, store as-is
            search_space[path] = hpo_config
    
    return search_space
