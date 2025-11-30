"""
Script to fix dashboard test logic.
Fixes TRACE_DATA assignment and update_resource_graph usage.
"""
from pathlib import Path
import re

filepath = "tests/visualization/test_dashboard_visualization.py"

def fix_test_logic():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Fix 1: TRACE_DATA assignment
    # Replace "from neural.dashboard.dashboard import TRACE_DATA" with "import neural.dashboard.dashboard as dashboard_module"
    # But wait, "from ... import update_trace_graph" is at the top.
    # The "from ... import TRACE_DATA" is inside the methods.
    
    new_content = content.replace(
        "from neural.dashboard.dashboard import TRACE_DATA",
        "import neural.dashboard.dashboard as dashboard_module"
    )
    
    new_content = new_content.replace(
        "TRACE_DATA = sample_trace_data",
        "dashboard_module.TRACE_DATA = sample_trace_data"
    )
    
    # Fix 2: update_resource_graph AttributeError
    # It might be because of how it's imported or patched.
    # Let's try to use dashboard_module.update_resource_graph instead of direct import if possible.
    # But direct import should work.
    # Maybe the test is patching 'neural.dashboard.dashboard.update_resource_graph' implicitly?
    # No, it patches 'neural.dashboard.dashboard.go.Figure' etc.
    
    # Let's try to fix the TRACE_DATA first, as that covers 7 failures.
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed test logic in: {filepath}")
    else:
        print(f"ℹ️ No changes needed in: {filepath}")

if __name__ == "__main__":
    fix_test_logic()
