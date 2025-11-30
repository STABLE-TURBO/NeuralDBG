"""
Script to fix dashboard visualization tests.
Replaces update_graph with update_trace_graph.
"""
from pathlib import Path

filepath = "tests/visualization/test_dashboard_visualization.py"

def fix_dashboard_tests():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Replace import
    # from neural.dashboard.dashboard import update_graph, update_resource_graph
    # to
    # from neural.dashboard.dashboard import update_trace_graph, update_resource_graph
    
    new_content = content.replace(
        "from neural.dashboard.dashboard import update_graph",
        "from neural.dashboard.dashboard import update_trace_graph"
    )
    
    # Replace calls
    # update_graph(1, ...) -> update_trace_graph(1, ...)
    new_content = new_content.replace("update_graph(", "update_trace_graph(")
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed dashboard tests in: {filepath}")
    else:
        print(f"ℹ️ No changes needed in: {filepath}")

if __name__ == "__main__":
    fix_dashboard_tests()
