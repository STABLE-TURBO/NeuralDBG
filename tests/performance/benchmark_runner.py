#!/usr/bin/env python
"""
Comprehensive benchmark runner for Neural performance tests.
"""
import time
import sys
import json
from pathlib import Path
import subprocess


class BenchmarkRunner:
    """Run and report on all performance benchmarks."""
    
    def __init__(self):
        self.results = {}
        self.failed = []
        
    def run_benchmark(self, test_file, test_name):
        """Run a single benchmark and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_file, '-v', '-s'],
                capture_output=True,
                text=True,
                timeout=120
            )
            elapsed = time.time() - start
            
            success = result.returncode == 0
            
            self.results[test_name] = {
                'success': success,
                'elapsed': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                print(f"✓ {test_name} completed in {elapsed:.2f}s")
            else:
                print(f"✗ {test_name} failed")
                self.failed.append(test_name)
                
            return success
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"✗ {test_name} timed out after {elapsed:.2f}s")
            self.failed.append(test_name)
            self.results[test_name] = {
                'success': False,
                'elapsed': elapsed,
                'error': 'timeout'
            }
            return False
            
    def run_all(self):
        """Run all benchmark suites."""
        benchmarks = [
            ('test_cli_startup.py', 'CLI Startup Benchmarks'),
            ('test_shape_propagation.py', 'Shape Propagation Benchmarks'),
            ('test_parser_performance.py', 'Parser Performance Benchmarks'),
            ('test_end_to_end.py', 'End-to-End Benchmarks'),
        ]
        
        print("\n" + "="*60)
        print("Neural Performance Benchmark Suite")
        print("="*60)
        
        test_dir = Path(__file__).parent
        
        for test_file, test_name in benchmarks:
            test_path = test_dir / test_file
            if test_path.exists():
                self.run_benchmark(str(test_path), test_name)
            else:
                print(f"⚠ Skipping {test_name}: file not found")
                
        self.print_summary()
        
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['success'])
        failed = total - passed
        
        print(f"\nTotal benchmarks: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if self.results:
            print("\nDetailed Results:")
            print("-" * 60)
            for name, result in self.results.items():
                status = "✓ PASS" if result['success'] else "✗ FAIL"
                elapsed = result.get('elapsed', 0)
                print(f"{status:8} {name:40} {elapsed:6.2f}s")
        
        if self.failed:
            print("\nFailed Benchmarks:")
            for name in self.failed:
                print(f"  - {name}")
                
        total_time = sum(r.get('elapsed', 0) for r in self.results.values())
        print(f"\nTotal benchmark time: {total_time:.2f}s")
        
        return failed == 0
        

def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
