"""
Script to fix update_resource_graph usage in dashboard tests.
"""
from pathlib import Path

filepath = "tests/visualization/test_dashboard_visualization.py"

def fix_resource_test():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Replace update_resource_graph(1) with dashboard_module.update_resource_graph(1)
    # This assumes dashboard_module is imported (which we did in fix_dashboard_test_logic.py)
    
    new_content = content.replace(
        "result = update_resource_graph(1)",
        "result = dashboard_module.update_resource_graph(1)"
    )
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed update_resource_graph usage in: {filepath}")
    else:
        print(f"ℹ️ No changes needed in: {filepath}")

if __name__ == "__main__":
    fix_resource_test()
