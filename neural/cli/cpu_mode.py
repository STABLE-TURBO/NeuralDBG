"""
CPU Mode Module for Neural DSL
This module provides functions to run Neural DSL in CPU-only mode.
"""

import os


def set_cpu_mode():
    """
    Set environment variables to force CPU mode.
    Must be called before importing deep learning frameworks.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NEURAL_FORCE_CPU'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_ENABLE_TENSOR_FLOAT_32_EXECUTION'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def is_cpu_mode() -> bool:
    """
    Check if CPU mode is enabled.
    """
    return (os.environ.get('NEURAL_FORCE_CPU', '').lower() in ['1', 'true', 'yes'] or
            os.environ.get('CUDA_VISIBLE_DEVICES', '') == '')

def run_with_cpu_mode(func, *args, **kwargs):
    """
    Run a function with CPU mode enabled temporarily.
    Restores original environment variables after execution.
    """
    original_env = {
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'NEURAL_FORCE_CPU': os.environ.get('NEURAL_FORCE_CPU', ''),
        'TF_ENABLE_ONEDNN_OPTS': os.environ.get('TF_ENABLE_ONEDNN_OPTS', ''),
        'TF_ENABLE_TENSOR_FLOAT_32_EXECUTION': os.environ.get('TF_ENABLE_TENSOR_FLOAT_32_EXECUTION', ''),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
        'TF_CPP_MIN_LOG_LEVEL': os.environ.get('TF_CPP_MIN_LOG_LEVEL', '')
    }

    try:
        set_cpu_mode()
        return func(*args, **kwargs)
    finally:
        for key, value in original_env.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

def get_pytorch_cpu_only():
    """
    Get PyTorch with CPU-only support.
    Ensures PyTorch is imported without CUDA support.
    """
    set_cpu_mode()
    import torch
    if hasattr(torch, 'cuda'):
        torch.cuda.is_available = lambda: False
    return torch
