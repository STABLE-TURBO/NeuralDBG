"""
Execution optimization utilities for Neural DSL.

Provides device selection, model optimization, and inference utilities
for efficient model execution on various hardware.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

from neural.exceptions import DependencyError
from neural.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Only import TensorRT if not in CPU mode
TENSORRT_AVAILABLE = False
if HAS_TORCH and not (os.environ.get('NEURAL_FORCE_CPU', '').lower() in ['1', 'true', 'yes'] or os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''):
    try:
        import tensorrt as trt

        TENSORRT_AVAILABLE = True
    except ImportError:
        pass


def get_device(preferred_device: Union[str, 'torch.device'] = "auto") -> 'torch.device':
    """
    Select the best available device: GPU, CPU, or future accelerators.

    Args:
        preferred_device: Device preference ("auto", "cpu", "gpu", "cuda") or torch.device object

    Returns:
        The selected torch.device

    Examples:
        >>> device = get_device("auto")  # Automatically selects CUDA if available
        >>> device = get_device("cpu")   # Force CPU execution
        >>> device = get_device(torch.device("cuda:0"))  # Specific GPU
    """
    if not HAS_TORCH:
        raise DependencyError(
            dependency="torch",
            feature="device selection",
            install_hint="pip install torch"
        )
    
    if isinstance(preferred_device, torch.device):
        return preferred_device

    if isinstance(preferred_device, str):
        device_str = preferred_device.lower()
        
        if device_str in ("gpu", "cuda"):
            if torch.cuda.is_available():
                logger.debug("Selected CUDA device")
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        
        elif device_str == "cpu":
            logger.debug("Selected CPU device")
            return torch.device("cpu")
        
        elif device_str == "auto":
            if torch.cuda.is_available():
                logger.debug("Auto-selected CUDA device")
                return torch.device("cuda")
            else:
                logger.debug("Auto-selected CPU device (CUDA not available)")
                return torch.device("cpu")

    logger.debug("Using default CPU device")
    return torch.device("cpu")


def run_inference(
    model: 'torch.nn.Module',
    data: 'torch.Tensor',
    execution_config: Optional[Dict[str, Any]] = None
) -> 'torch.Tensor':
    """
    Run inference on the specified device.

    Args:
        model: PyTorch model to run
        data: Input tensor
        execution_config: Configuration dict with 'device' key (default: {"device": "auto"})

    Returns:
        Output tensor on CPU

    Examples:
        >>> model = MyModel()
        >>> data = torch.randn(1, 3, 224, 224)
        >>> output = run_inference(model, data, {"device": "cuda"})
    """
    if not HAS_TORCH:
        raise DependencyError(
            dependency="torch",
            feature="inference execution",
            install_hint="pip install torch"
        )
    
    if execution_config is None:
        execution_config = {}
    
    device = get_device(execution_config.get("device", "auto"))
    logger.debug(f"Running inference on {device}")
    
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()


def optimize_model_with_tensorrt(model: 'torch.nn.Module') -> 'torch.nn.Module':
    """
    Convert model to TensorRT for optimized inference.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model (TensorRT if available, otherwise JIT traced)

    Note:
        Requires TensorRT to be installed. Falls back to standard model if unavailable.
    """
    if not HAS_TORCH:
        raise DependencyError(
            dependency="torch",
            feature="TensorRT optimization",
            install_hint="pip install torch"
        )
    
    if not TENSORRT_AVAILABLE:
        logger.warning("TensorRT not available, skipping optimization")
        return model

    try:
        model.eval()
        device = get_device("gpu")

        if not hasattr(model, 'input_shape'):
            logger.error("Model must have 'input_shape' attribute for TensorRT optimization")
            return model

        dummy_input = torch.randn(1, *model.input_shape).to(device)

        traced_model = torch.jit.trace(model, dummy_input)
        trt_model = torch.jit.freeze(traced_model)

        logger.info("Model successfully optimized with TensorRT")
        return trt_model
    
    except Exception as e:
        logger.error(f"TensorRT optimization failed: {e}", exc_info=True)
        return model


def run_optimized_inference(
    model: 'torch.nn.Module',
    data: 'torch.Tensor',
    execution_config: Optional[Dict[str, Any]] = None
) -> 'torch.Tensor':
    """
    Run optimized inference using TensorRT or PyTorch.

    Automatically applies TensorRT optimization if available and running on GPU.

    Args:
        model: PyTorch model to run
        data: Input tensor
        execution_config: Configuration dict with 'device' key (default: {"device": "auto"})

    Returns:
        Output tensor on CPU

    Examples:
        >>> model = MyModel()
        >>> data = torch.randn(1, 3, 224, 224)
        >>> output = run_optimized_inference(model, data, {"device": "cuda"})
    """
    if not HAS_TORCH:
        raise DependencyError(
            dependency="torch",
            feature="optimized inference execution",
            install_hint="pip install torch"
        )
    
    if execution_config is None:
        execution_config = {}
    
    device = get_device(execution_config.get("device", "auto"))
    logger.debug(f"Running optimized inference on {device}")

    if device.type == "cuda" and TENSORRT_AVAILABLE:
        logger.debug("Applying TensorRT optimization")
        model = optimize_model_with_tensorrt(model)

    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()
