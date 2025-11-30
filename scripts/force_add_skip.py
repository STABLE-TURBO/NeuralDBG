"""
Script to force add skip_if_no_cuda definition.
"""
from pathlib import Path

files = [
    "tests/test_device_execution.py",
    "tests/integration_tests/test_device_integration.py"
]

SKIP_DEF = """
import pytest
import torch

# Skip GPU tests if CUDA not available
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
"""

def force_add(filepath):
    path = Path(filepath)
    if not path.exists():
        return
    
    content = path.read_text(encoding='utf-8')
    
    # Remove existing definition if partial/broken
    if "skip_if_no_cuda =" not in content:
        # Prepend to top after imports? Or just at top?
        # Let's put it after the first few lines of imports if possible, or just at the top.
        # Safest is to put it after imports.
        
        lines = content.splitlines()
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import_idx = i
        
        # Insert after last import
        lines.insert(last_import_idx + 1, SKIP_DEF)
        new_content = "\n".join(lines)
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Added definition to {filepath}")
    else:
        print(f"ℹ️ Definition already exists in {filepath}")

if __name__ == "__main__":
    for f in files:
        force_add(f)
