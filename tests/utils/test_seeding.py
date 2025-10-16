"""
Tests for seeding utilities to ensure deterministic behavior.
"""

import pytest
import numpy as np
from neural.utils.seeding import set_global_seed, get_current_seed

def test_set_global_seed_deterministic():
    """Test that setting the same seed produces the same random numbers."""
    # Set a specific seed
    seed = 42
    set_global_seed(seed)
    
    # Generate some random numbers
    np_random1 = np.random.rand(10)
    
    # Reset seed and generate again
    set_global_seed(seed)
    np_random2 = np.random.rand(10)
    
    # Arrays should be exactly equal
    np.testing.assert_array_equal(np_random1, np_random2)

def test_get_current_seed():
    """Test that get_current_seed returns a valid seed value."""
    # Set a specific seed
    seed = 12345
    set_global_seed(seed)
    
    # Get current seed
    current_seed = get_current_seed()
    assert isinstance(current_seed, int)
    
def test_random_seed_generation():
    """Test that None seed generates different random sequences."""
    # Set None seed twice
    seed1 = set_global_seed(None)
    rand1 = np.random.rand()
    
    seed2 = set_global_seed(None)
    rand2 = np.random.rand()
    
    # Seeds and random numbers should be different
    assert seed1 != seed2
    assert rand1 != rand2