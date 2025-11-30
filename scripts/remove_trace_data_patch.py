"""
Script to remove TRACE_DATA patch from dashboard tests.
"""
from pathlib import Path
import re

filepath = "tests/visualization/test_dashboard_visualization.py"

def remove_patch():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Remove lines containing @patch('neural.dashboard.dashboard.TRACE_DATA'
    # Use regex to match the whole line including indentation
    
    pattern = r"^\s*@patch\('neural\.dashboard\.dashboard\.TRACE_DATA'.*$\n"
    
    new_content = re.sub(pattern, "", content, flags=re.MULTILINE)
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Removed TRACE_DATA patches in: {filepath}")
    else:
        print(f"ℹ️ No changes needed in: {filepath}")

if __name__ == "__main__":
    remove_patch()
