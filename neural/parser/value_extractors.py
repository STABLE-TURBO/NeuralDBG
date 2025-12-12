"""Value extraction utilities for Neural DSL parser.

This module contains utilities for extracting Python values from parse tree nodes.
"""

from typing import Any, Dict, List, Tuple, Optional
from lark import Token, Tree


def extract_token_value(token: Token) -> Any:
    """Extract value from a Lark token.
    
    Args:
        token: Lark Token object
        
    Returns:
        Extracted Python value
    """
    if token.type == 'NAME':
        return token.value
    if token.type in ('INT', 'FLOAT', 'NUMBER', 'SIGNED_NUMBER'):
        try:
            return int(token.value)
        except ValueError:
            return float(token.value)
    elif token.type == 'BOOL':
        return token.value.lower() == 'true'
    elif token.type == 'STRING':
        return token.value.strip('"')
    elif token.type == 'WS_INLINE':
        return token.value.strip()
    return token.value


def extract_tree_value(tree: Tree, extract_value_fn) -> Any:
    """Extract value from a Lark tree node.
    
    Args:
        tree: Lark Tree object
        extract_value_fn: Function to recursively extract values
        
    Returns:
        Extracted Python value
    """
    if tree.data == 'number_or_none':
        child = tree.children[0]
        if isinstance(child, Token) and child.value.upper() in ('NONE', 'None'):
            return None
        else:
            return extract_value_fn(child)
    elif tree.data == 'string_value':
        return extract_value_fn(tree.children[0])
    elif tree.data == 'number':
        return extract_value_fn(tree.children[0])
    elif tree.data == 'bool_value':
        return extract_value_fn(tree.children[0])
    elif tree.data == 'params':
        return [extract_value_fn(child) for child in tree.children]
    elif tree.data in ('tuple_', 'explicit_tuple'):
        return tuple(extract_value_fn(child) for child in tree.children)
    else:
        # Generic tree processing
        extracted = [extract_value_fn(child) for child in tree.children]
        
        # Check if we should return as dict
        if any(isinstance(e, dict) for e in extracted):
            return extracted
            
        # Try to form a dictionary from key-value pairs
        if len(tree.children) % 2 == 0:
            try:
                valid = True
                pairs = []
                for k_node, v_node in zip(tree.children[::2], tree.children[1::2]):
                    key = extract_value_fn(k_node)
                    if not isinstance(key, str):
                        valid = False
                        break
                    value = extract_value_fn(v_node)
                    pairs.append((key, value))
                if valid:
                    return dict(pairs)
                else:
                    return extracted
            except TypeError:
                return extracted
        else:
            return extracted


def extract_value_recursive(item: Any, extract_value_fn) -> Any:
    """Recursively extract values from various data structures.
    
    Args:
        item: Item to extract value from (Token, Tree, list, dict, or primitive)
        extract_value_fn: Function to recursively extract values
        
    Returns:
        Extracted Python value
    """
    if isinstance(item, Token):
        return extract_token_value(item)
    elif isinstance(item, Tree):
        return extract_tree_value(item, extract_value_fn)
    elif isinstance(item, list):
        return [extract_value_fn(elem) for elem in item]
    elif isinstance(item, dict):
        return {k: extract_value_fn(v) for k, v in item.items()}
    return item


def shift_if_token(items: List[Any]) -> List[Any]:
    """Remove first item if it's a Token (for alias rule calls).
    
    Args:
        items: List of items from parse tree
        
    Returns:
        Modified list with first token removed if applicable
    """
    if items and isinstance(items[0], Token):
        return items[1:]
    return items


def extract_named_input_shapes(items: List[Any], extract_value_fn) -> Dict[str, Any]:
    """Extract named input shapes from parse tree items.
    
    Args:
        items: Parse tree items (alternating NAME and shape nodes)
        extract_value_fn: Function to extract values
        
    Returns:
        Dictionary mapping input names to shapes
    """
    result = {}
    i = 0
    while i < len(items):
        name_token = items[i]
        shape_node = items[i+1] if i + 1 < len(items) else None
        name = name_token.value if hasattr(name_token, 'value') else str(name_token)
        shape = extract_value_fn(shape_node)
        
        # Normalize 1D shapes: (10,) -> 10
        if isinstance(shape, tuple) and len(shape) == 1:
            shape = shape[0]
        result[name] = shape
        i += 2
    return result


def extract_branch_sublayers(items: List[Any], extract_value_fn) -> List[Any]:
    """Extract sublayers from branch specification items.
    
    Args:
        items: Parse tree items (first is branch name, rest are sublayers)
        extract_value_fn: Function to extract values
        
    Returns:
        List of sublayer configurations
    """
    sub_layers = []
    # Skip first item (branch name) and collect rest as sublayers
    for child in items[1:]:
        val = extract_value_fn(child)
        if isinstance(val, list):
            sub_layers.extend(val)
        else:
            sub_layers.append(val)
    return sub_layers


def merge_param_list(param_values: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Merge a list of parameters into ordered and named parameters.
    
    This is used for layers that accept mixed positional and named parameters.
    
    Args:
        param_values: List of parameter values
        
    Returns:
        Tuple of (ordered_params, named_params)
    """
    ordered_params = []
    named_params = {}
    
    for val in param_values:
        if isinstance(val, dict):
            if 'hpo' in val:
                # Handle HPO expressions
                if len(named_params) == 0 and len(ordered_params) == 0:
                    ordered_params.append(val)
                else:
                    named_params.update(val)
            else:
                named_params.update(val)
        elif isinstance(val, list):
            ordered_params.extend(val)
        else:
            ordered_params.append(val)
            
    return ordered_params, named_params


def validate_param_count(ordered_params: List[Any], max_count: int, layer_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that the number of positional parameters doesn't exceed a maximum.
    
    Args:
        ordered_params: List of positional parameters
        max_count: Maximum allowed count
        layer_type: Type of the layer (for error message)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(ordered_params) > max_count:
        return False, f"{layer_type} accepts at most {max_count} positional arguments"
    return True, None
