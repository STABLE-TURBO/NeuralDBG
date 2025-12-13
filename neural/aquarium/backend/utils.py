"""Utility functions for the backend bridge."""

import hashlib
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def hash_code(code: str) -> str:
    """Generate a hash of the code for caching/identification.

    Args:
        code: Python code string

    Returns:
        Hash string
    """
    return hashlib.sha256(code.encode()).hexdigest()


def serialize_model_data(model_data: Dict[str, Any]) -> str:
    """Serialize model data to JSON string.

    Args:
        model_data: Model data dictionary

    Returns:
        JSON string
    """
    try:
        return json.dumps(model_data, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to serialize model data: {e}", exc_info=True)
        return "{}"


def validate_backend(backend: str) -> bool:
    """Validate if backend is supported.

    Args:
        backend: Backend name

    Returns:
        True if valid, False otherwise
    """
    return backend.lower() in ["tensorflow", "pytorch", "onnx"]


def validate_parser_type(parser_type: str) -> bool:
    """Validate if parser type is supported.

    Args:
        parser_type: Parser type

    Returns:
        True if valid, False otherwise
    """
    return parser_type.lower() in ["network", "research"]


def format_error_message(error: Exception) -> str:
    """Format exception into a user-friendly error message.

    Args:
        error: Exception object

    Returns:
        Formatted error string
    """
    error_type = type(error).__name__
    error_message = str(error)
    return f"{error_type}: {error_message}"


def extract_layer_types(model_data: Dict[str, Any]) -> list:
    """Extract all layer types from model data.

    Args:
        model_data: Model data dictionary

    Returns:
        List of layer type strings
    """
    layer_types = []
    for layer in model_data.get("layers", []):
        if isinstance(layer, dict) and "type" in layer:
            layer_types.append(layer["type"])
    return layer_types


def count_parameters(model_data: Dict[str, Any]) -> int:
    """Estimate total parameter count in the model.

    Args:
        model_data: Model data dictionary

    Returns:
        Estimated parameter count
    """
    total_params = 0

    for layer in model_data.get("layers", []):
        if not isinstance(layer, dict) or "type" not in layer:
            continue

        layer_type = layer["type"]
        params = layer.get("params", {})

        if layer_type == "Dense":
            units = params.get("units", 0)
            total_params += units

        elif layer_type in ["Conv2D", "Conv1D", "Conv3D"]:
            filters = params.get("filters", 0)
            kernel_size = params.get("kernel_size", 3)
            if isinstance(kernel_size, (list, tuple)):
                kernel_size = kernel_size[0]
            total_params += filters * (kernel_size ** 2)

        elif layer_type in ["LSTM", "GRU"]:
            units = params.get("units", 0)
            total_params += units * 4

    return total_params
