"""
Validation script for HPO examples.

This script validates that HPO examples can be parsed and compiled correctly.
"""

import os
import sys


# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural.parser.parser import ModelTransformer


def validate_hpo_example(filepath: str) -> bool:
    """
    Validate that an HPO example can be parsed correctly.
    
    Args:
        filepath: Path to the .neural file
        
    Returns:
        True if validation succeeds, False otherwise
    """
    print(f"\nValidating: {filepath}")
    print("-" * 60)
    
    try:
        # Read the file
        with open(filepath, 'r') as f:
            config = f.read()
        
        # Parse the configuration
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(config)
        
        # Validate that we got a model dict
        assert 'input' in model_dict, "Missing 'input' in model_dict"
        assert 'layers' in model_dict, "Missing 'layers' in model_dict"
        assert isinstance(hpo_params, list), "hpo_params should be a list"
        
        # Print summary
        print("✓ Successfully parsed")
        print(f"  - Input shape: {model_dict['input']['shape']}")
        print(f"  - Number of layers: {len(model_dict['layers'])}")
        print(f"  - HPO parameters found: {len(hpo_params)}")
        
        if hpo_params:
            print("  - HPO details:")
            for i, param in enumerate(hpo_params):
                hpo_info = param.get('hpo', {})
                print(f"    {i+1}. Type: {hpo_info.get('type')}, "
                      f"Layer: {param.get('layer_type', 'N/A')}, "
                      f"Param: {param.get('param_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("="*60)
    print("HPO Examples Validation")
    print("="*60)
    
    # List of HPO examples to validate
    examples = [
        'mnist_hpo.neural',
        'advanced_hpo_v0.2.5.neural',
    ]
    
    results = {}
    for example in examples:
        filepath = os.path.join(os.path.dirname(__file__), example)
        if os.path.exists(filepath):
            results[example] = validate_hpo_example(filepath)
        else:
            print(f"\n✗ File not found: {example}")
            results[example] = False
    
    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for example, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {example}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
