"""
Utilities for deterministic seeding across frameworks and random number generators.
"""

import random
import numpy as np
import os
from typing import Optional

def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set deterministic seed for all supported frameworks and random number generators.
    
    Args:
        seed: Integer seed value. If None, generates a random seed.
        
    Returns:
        The seed value used
    """
    if seed is None:
        # Generate random seed between 0 and 2^32 - 1
        seed = random.randint(0, 2**32 - 1)
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set environment variable for TensorFlow determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Lazy import TensorFlow and PyTorch to avoid unnecessary imports
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
        
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
        
    return seed

def get_current_seed() -> int:
    """
    Get the current random seed value from NumPy.
    
    Returns:
        Current seed value
    """
    state = np.random.get_state()
    if isinstance(state, tuple) and len(state) > 1 and isinstance(state[1], np.ndarray):
        return int(state[1][0])
    return 0  # Fallback value if state structure is unexpected