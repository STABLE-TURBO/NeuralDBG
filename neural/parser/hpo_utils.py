"""HPO (Hyperparameter Optimization) utilities for Neural DSL parser.

This module contains helper functions for processing HPO expressions and parameters.
"""

from enum import Enum
import re
from typing import Any, Callable, Dict, List, Optional


class Severity(Enum):
    """Severity levels for logging and error reporting."""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


def extract_hpo_expressions(text: str) -> List[str]:
    """Extract HPO expressions from text, handling nested parentheses.
    
    Args:
        text: Text containing HPO expressions
        
    Returns:
        List of HPO expression contents (without 'HPO(' and ')')
    """
    results = []
    start_idx = text.find('HPO(')
    while start_idx != -1:
        # Find the matching closing parenthesis
        paren_level = 0
        for i in range(start_idx + 4, len(text)):  # Skip 'HPO('
            if text[i] == '(':
                paren_level += 1
            elif text[i] == ')':
                if paren_level == 0:
                    # Found the matching closing parenthesis
                    results.append(text[start_idx + 4:i])
                    break
                paren_level -= 1
        # Find the next HPO expression
        start_idx = text.find('HPO(', start_idx + 1)
    return results


def parse_log_range_hpo(hpo_expr: str) -> Optional[Dict[str, Any]]:
    """Parse a log_range HPO expression.
    
    Args:
        hpo_expr: HPO expression string
        
    Returns:
        Dictionary with HPO configuration or None if not a log_range expression
    """
    match = re.search(r'log_range\(([^,]+),\s*([^)]+)\)', hpo_expr)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return {'type': 'log_range', 'min': low, 'max': high}
    return None


def parse_range_hpo(hpo_expr: str) -> Optional[Dict[str, Any]]:
    """Parse a range HPO expression.
    
    Args:
        hpo_expr: HPO expression string
        
    Returns:
        Dictionary with HPO configuration or None if not a range expression
    """
    match = re.search(r'range\(([^,]+),\s*([^,]+)(?:,\s*step=([^)]+))?\)', hpo_expr)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        hpo_dict = {'type': 'range', 'start': low, 'end': high}
        step = match.group(3)
        if step:
            hpo_dict['step'] = float(step)
        return hpo_dict
    return None


def parse_choice_hpo(hpo_expr: str) -> Optional[Dict[str, Any]]:
    """Parse a choice/categorical HPO expression.
    
    Args:
        hpo_expr: HPO expression string
        
    Returns:
        Dictionary with HPO configuration or None if not a choice expression
    """
    match = re.search(r'choice\(([^)]+)\)', hpo_expr)
    if match:
        choices_str = match.group(1)
        # Handle different types of choices (numbers, strings)
        try:
            choices = [float(x.strip()) for x in choices_str.split(',')]
        except ValueError:
            # Handle string choices
            choices = [x.strip().strip('"\'') for x in choices_str.split(',')]
        return {'type': 'categorical', 'values': choices}
    return None


def parse_hpo_expression(hpo_expr: str) -> Optional[Dict[str, Any]]:
    """Parse an HPO expression and return its configuration.
    
    Args:
        hpo_expr: HPO expression string
        
    Returns:
        Dictionary with HPO configuration or None if parsing failed
    """
    # Try each HPO type
    if 'log_range' in hpo_expr:
        return parse_log_range_hpo(hpo_expr)
    elif 'range' in hpo_expr:
        return parse_range_hpo(hpo_expr)
    elif 'choice' in hpo_expr:
        return parse_choice_hpo(hpo_expr)
    return None


def track_hpo_in_optimizer_string(optimizer_info: str, track_hpo_fn: Callable) -> None:
    """Track HPO parameters in an optimizer string.
    
    Args:
        optimizer_info: Optimizer string that may contain HPO expressions
        track_hpo_fn: Function to call for tracking HPO parameters
    """
    if 'HPO(' not in optimizer_info:
        return

    hpo_matches = extract_hpo_expressions(optimizer_info)

    for hpo_expr in hpo_matches:
        hpo_dict = parse_hpo_expression(hpo_expr)
        if hpo_dict:
            hpo_param = {'hpo': hpo_dict}
            track_hpo_fn('optimizer', 'learning_rate', hpo_param, None)


def track_hpo_in_optimizer_params(params: Dict[str, Any], track_hpo_fn: Callable) -> None:
    """Track HPO parameters in optimizer parameters.
    
    Args:
        params: Optimizer parameters dictionary
        track_hpo_fn: Function to call for tracking HPO parameters
    """
    for param_name, param_value in params.items():
        if isinstance(param_value, dict) and 'hpo' in param_value:
            track_hpo_fn('optimizer', param_name, param_value, None)
        # Track HPO parameters in learning rate schedules
        elif param_name == 'learning_rate' and isinstance(param_value, dict):
            if 'type' in param_value and 'args' in param_value:
                for i, arg in enumerate(param_value['args']):
                    if isinstance(arg, dict) and 'hpo' in arg:
                        track_hpo_fn('optimizer', f'learning_rate.args[{i}]', arg, None)


def track_hpo_in_lr_schedule_string(lr_value: str, track_hpo_fn: Callable) -> None:
    """Track HPO parameters in a learning rate schedule string.
    
    Args:
        lr_value: Learning rate schedule string
        track_hpo_fn: Function to call for tracking HPO parameters
    """
    if 'HPO(' not in lr_value:
        return

    hpo_matches = re.findall(r'HPO\((.*?)\)', lr_value)

    for hpo_expr in hpo_matches:
        hpo_dict = parse_hpo_expression(hpo_expr)
        if hpo_dict:
            track_hpo_fn('optimizer', 'learning_rate', {'hpo': hpo_dict}, None)


def has_hpo_parameter(param_value: Any) -> bool:
    """Check if a parameter value contains an HPO specification.
    
    Args:
        param_value: Parameter value to check
        
    Returns:
        True if the value contains HPO specification
    """
    if isinstance(param_value, dict) and 'hpo' in param_value:
        return True
    if isinstance(param_value, list) and len(param_value) > 0:
        if isinstance(param_value[0], dict) and 'hpo' in param_value[0]:
            return True
    return False


def extract_hpo_from_list(param_value: List[Any]) -> Optional[Dict[str, Any]]:
    """Extract HPO configuration from a list parameter.
    
    Args:
        param_value: List that may contain HPO configuration
        
    Returns:
        HPO configuration dictionary or None
    """
    if len(param_value) > 0 and isinstance(param_value[0], dict) and 'hpo' in param_value[0]:
        return param_value[0]
    return None


def create_categorical_hpo_from_list(values: List[Any]) -> Dict[str, Any]:
    """Create a categorical HPO configuration from a list of values.
    
    Args:
        values: List of possible values
        
    Returns:
        HPO configuration dictionary
    """
    return {
        'hpo': {
            'type': 'categorical',
            'values': values,
            'original_values': [str(v) for v in values]
        }
    }
