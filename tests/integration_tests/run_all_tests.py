#!/usr/bin/env python
"""
Convenience script to run all integration tests with various configurations.

Usage:
    python tests/integration_tests/run_all_tests.py [options]

Options:
    --fast          Run only fast tests (skip execution tests)
    --backend NAME  Run tests for specific backend (pytorch, tensorflow, onnx)
    --feature NAME  Run tests for specific feature (hpo, tracking, shape)
    --coverage      Generate coverage report
    --verbose       Verbose output
    --parallel      Run tests in parallel
    --help          Show this help message
"""

import sys
import os
import subprocess
import argparse


def run_pytest(args_list):
    """Run pytest with given arguments."""
    cmd = ['pytest'] + args_list
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run Neural DSL integration tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (skip slow execution tests)')
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'onnx'],
                       help='Run tests for specific backend')
    parser.add_argument('--feature', choices=['hpo', 'tracking', 'shape', 'parsing', 'execution'],
                       help='Run tests for specific feature')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--parallel', '-n', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    parser.add_argument('--test', type=str,
                       help='Run specific test (format: file::class::test_name)')
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = ['tests/integration_tests/']
    
    # Add verbosity
    if args.verbose:
        pytest_args.append('-v')
    else:
        pytest_args.append('-v')  # Always verbose for integration tests
    
    # Add coverage
    if args.coverage:
        pytest_args.extend(['--cov=neural', '--cov-report=term', '--cov-report=html'])
    
    # Add parallel execution
    if args.parallel:
        pytest_args.extend(['-n', 'auto'])
    
    # Filter by speed
    if args.fast:
        pytest_args.extend(['-m', 'not slow'])
    
    # Filter by backend
    if args.backend:
        pytest_args.extend(['-k', args.backend])
    
    # Filter by feature
    if args.feature:
        pytest_args.extend(['-k', args.feature])
    
    # Run specific file
    if args.file:
        pytest_args = [f'tests/integration_tests/{args.file}'] + pytest_args[1:]
    
    # Run specific test
    if args.test:
        pytest_args = [f'tests/integration_tests/{args.test}'] + pytest_args[1:]
    
    # Run tests
    print("=" * 80)
    print("Neural DSL Integration Tests")
    print("=" * 80)
    print()
    
    return_code = run_pytest(pytest_args)
    
    print()
    print("=" * 80)
    if return_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with code {return_code}")
    print("=" * 80)
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())
