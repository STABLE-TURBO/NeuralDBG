"""
Script to fix import paths in device tests.
"""
from pathlib import Path

files_to_fix = [
    "tests/test_device_execution.py",
    "tests/integration_tests/test_device_integration.py"
]

def fix_imports(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Fix 1: execution_optimization -> neural.execution_optimization
    new_content = content.replace("from execution_optimization", "from neural.execution_optimization")
    
    # Fix 2: If there are other similar issues, fix them too
    # Example: from pretrained import -> from neural.pretrained import
    # But let's stick to what we saw in the error first.

    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed imports in: {filepath}")
    else:
        print(f"ℹ️ No changes needed in: {filepath}")

if __name__ == "__main__":
    for f in files_to_fix:
        fix_imports(f)
