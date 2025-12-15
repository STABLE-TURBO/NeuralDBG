"""
Basic test script for HPO functionality.

This script tests that the HPO module can:
1. Parse HPO configurations
2. Create dynamic models
3. Resolve HPO parameters
4. Handle edge cases properly
"""

import os
import sys


# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from neural.hpo.hpo import create_dynamic_model, resolve_hpo_params
from neural.parser.parser import ModelTransformer


class MockTrial:
    """Mock Optuna trial for testing."""
    def suggest_categorical(self, name, choices):
        return choices[0]
    
    def suggest_float(self, name, low, high, step=None, log=False):
        return low if not log else low * 1.5
    
    def suggest_int(self, name, low, high, step=None, log=False):
        return low


def test_parse_simple_hpo():
    """Test parsing a simple HPO configuration."""
    print("\nTest 1: Parse Simple HPO Configuration")
    print("-" * 60)
    
    config = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(HPO(choice(64, 128)))
            Output(10)
        optimizer: Adam(learning_rate=0.001)
    }
    """
    
    try:
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        assert 'input' in model_dict
        assert 'layers' in model_dict
        assert len(hpo_params) >= 1
        
        print("✓ Successfully parsed HPO configuration")
        print(f"  - Found {len(hpo_params)} HPO parameter(s)")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_dynamic_model():
    """Test creating a dynamic model with HPO parameters."""
    print("\nTest 2: Create Dynamic Model")
    print("-" * 60)
    
    config = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(HPO(choice(64, 128)))
            Dropout(HPO(range(0.3, 0.5, step=0.1)))
            Output(10)
        optimizer: Adam(learning_rate=0.001)
    }
    """
    
    try:
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)  # NCHW format
        output = model(x)
        
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        
        print("✓ Successfully created dynamic model")
        print(f"  - Model has {len(list(model.parameters()))} parameter tensors")
        print(f"  - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_resolve_hpo_params():
    """Test resolving HPO parameters."""
    print("\nTest 3: Resolve HPO Parameters")
    print("-" * 60)
    
    config = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(HPO(choice(64, 128)))
            Output(10)
        optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
    }
    """
    
    try:
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        trial = MockTrial()
        resolved_dict = resolve_hpo_params(model_dict, trial, hpo_params)
        
        # Check that HPO parameters were resolved
        dense_layer = resolved_dict['layers'][1]  # Dense is second layer after Flatten
        assert 'units' in dense_layer['params']
        assert isinstance(dense_layer['params']['units'], (int, float))
        
        # Check optimizer learning rate
        if resolved_dict.get('optimizer') and resolved_dict['optimizer'].get('params'):
            lr = resolved_dict['optimizer']['params'].get('learning_rate')
            if lr is not None:
                assert isinstance(lr, (int, float))
        
        print("✓ Successfully resolved HPO parameters")
        print(f"  - Dense units: {dense_layer['params']['units']}")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_hpo_edge_cases():
    """Test HPO edge cases."""
    print("\nTest 4: HPO Edge Cases")
    print("-" * 60)
    
    # Test with minimal network
    config = """
    network MinimalNet {
        input: (10,)
        layers:
            Output(5)
        optimizer: Adam(learning_rate=0.001)
    }
    """
    
    try:
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        # Test with 1D input
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
        
        print("✓ Successfully handled edge case (minimal network)")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_hpo_types():
    """Test multiple HPO parameter types."""
    print("\nTest 5: Multiple HPO Parameter Types")
    print("-" * 60)
    
    config = """
    network MultiHPONet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(HPO(choice(64, 128, 256)))
            Dropout(HPO(range(0.3, 0.7, step=0.1)))
            Dense(HPO(choice(32, 64)))
            Output(10)
        optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
    }
    """
    
    try:
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        # Check that we found multiple HPO parameters
        assert len(hpo_params) >= 3, f"Expected at least 3 HPO params, found {len(hpo_params)}"
        
        # Check HPO types
        hpo_types = [param['hpo']['type'] for param in hpo_params]
        assert 'categorical' in hpo_types, "Should have categorical HPO"
        has_range = 'range' in hpo_types or 'log_range' in hpo_types
        assert has_range, "Should have range or log_range HPO"
        
        # Create model
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        
        print("✓ Successfully handled multiple HPO types")
        print(f"  - Found {len(hpo_params)} HPO parameters")
        print(f"  - HPO types: {set(hpo_types)}")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("HPO Module Basic Tests")
    print("="*60)
    
    tests = [
        ("Parse Simple HPO", test_parse_simple_hpo),
        ("Create Dynamic Model", test_create_dynamic_model),
        ("Resolve HPO Params", test_resolve_hpo_params),
        ("HPO Edge Cases", test_hpo_edge_cases),
        ("Multiple HPO Types", test_multiple_hpo_types),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
