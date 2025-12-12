"""
Benchmark CLI startup time with lazy imports.
"""
import time
import subprocess
import sys
import pytest


def test_cli_import_time():
    """Test that CLI imports are lazy and fast."""
    start = time.time()
    from neural.cli import cli
    import_time = time.time() - start
    
    assert import_time < 2.0, f"CLI import took {import_time:.3f}s, expected < 2.0s"
    print(f"\n✓ CLI import time: {import_time:.3f}s")


def test_cli_help_time():
    """Test CLI --help performance."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, '-m', 'neural.cli.cli', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    help_time = time.time() - start
    
    assert result.returncode == 0 or result.returncode == 2
    assert help_time < 5.0, f"CLI --help took {help_time:.3f}s, expected < 5.0s"
    print(f"✓ CLI --help time: {help_time:.3f}s")


def test_lazy_imports_not_loaded():
    """Verify heavy dependencies aren't loaded on import."""
    import neural.cli.lazy_imports as lazy_imports
    
    tf_loader = lazy_imports.tensorflow
    torch_loader = lazy_imports.torch
    
    assert tf_loader.module is None, "TensorFlow was loaded eagerly"
    assert torch_loader.module is None, "PyTorch was loaded eagerly"
    print("✓ Heavy dependencies remain unloaded")


def test_module_cache():
    """Test that module cache works correctly."""
    from neural.cli.lazy_imports import lazy_import, _module_cache
    
    initial_cache_size = len(_module_cache)
    
    loader1 = lazy_import('json')
    _ = loader1.dumps
    
    loader2 = lazy_import('json')
    _ = loader2.loads
    
    cache_hits = len(_module_cache) - initial_cache_size
    assert cache_hits == 1, "Module cache should reuse loaded modules"
    print(f"✓ Module cache working (cache size: {len(_module_cache)})")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
