"""
Script to fix Cloud Executor tests.
1. Fix subprocess.CalledProcessError mocking
2. Fix ngrok URL expectations
"""
from pathlib import Path
import re

filepath = "tests/cloud/test_cloud_executor.py"

def fix_cloud_executor_tests():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # 1. Import subprocess
    if "import subprocess" not in content:
        content = content.replace("import unittest", "import unittest\nimport subprocess")
    
    # 2. Fix CalledProcessError in setUp
    # Find self.mock_subprocess = self.subprocess_patcher.start()
    # Add self.mock_subprocess.CalledProcessError = subprocess.CalledProcessError
    
    setup_pattern = r"(self\.mock_subprocess = self\.subprocess_patcher\.start\(\))"
    if "self.mock_subprocess.CalledProcessError" not in content:
        content = re.sub(
            setup_pattern,
            r"\1\n        self.mock_subprocess.CalledProcessError = subprocess.CalledProcessError",
            content
        )
    
    # 3. Fix ngrok URL expectations
    # Replace "http://localhost:8050" with "https://test.ngrok.io" in test_start_debug_dashboard
    # Replace "http://localhost:8051" with "https://test.ngrok.io" in test_start_nocode_interface
    
    # Be careful not to replace the default value in the code, only in assertions
    # The assertions look like: self.assertEqual(result['dashboard_url'], "http://localhost:8050")
    
    content = content.replace(
        'self.assertEqual(result[\'dashboard_url\'], "http://localhost:8050")',
        'self.assertEqual(result[\'dashboard_url\'], "https://test.ngrok.io")'
    )
    
    content = content.replace(
        'self.assertEqual(result[\'interface_url\'], "http://localhost:8051")',
        'self.assertEqual(result[\'interface_url\'], "https://test.ngrok.io")'
    )
    
    path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed Cloud Executor tests in: {filepath}")

if __name__ == "__main__":
    fix_cloud_executor_tests()
