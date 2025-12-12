#!/usr/bin/env python
"""
Validate all Neural DSL examples by parsing and compiling them.
This script is used in CI to ensure all examples are syntactically correct.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code


def validate_neural_file(filepath):
    """Validate a single .neural file"""
    print(f"\nValidating: {filepath}")
    
    try:
        # Read file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse
        parser = create_parser(start_rule='network')
        tree = parser.parse(content)
        
        # Transform
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        print(f"  ✓ Parsed successfully")
        print(f"  ✓ Model name: {model_data.get('name', 'Unknown')}")
        print(f"  ✓ Layers: {len(model_data.get('layers', []))}")
        
        # Try compiling to TensorFlow
        try:
            code = generate_code(model_data, 'tensorflow')
            print(f"  ✓ Compiled to TensorFlow ({len(code)} bytes)")
        except Exception as e:
            print(f"  ⚠ TensorFlow compilation warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Validate all .neural files in use_cases directory"""
    script_dir = Path(__file__).parent
    neural_files = list(script_dir.glob('*.neural'))
    
    if not neural_files:
        print("No .neural files found!")
        sys.exit(1)
    
    print(f"Found {len(neural_files)} .neural files to validate")
    print("=" * 60)
    
    results = []
    for filepath in sorted(neural_files):
        success = validate_neural_file(filepath)
        results.append((filepath.name, success))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for filename, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {filename}")
    
    failed_count = sum(1 for _, success in results if not success)
    passed_count = len(results) - failed_count
    
    print(f"\nTotal: {len(results)} | Passed: {passed_count} | Failed: {failed_count}")
    
    if failed_count > 0:
        print("\n❌ Validation failed!")
        sys.exit(1)
    else:
        print("\n✅ All examples validated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
