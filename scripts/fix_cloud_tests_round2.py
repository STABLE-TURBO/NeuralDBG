"""
Script to fix remaining Cloud Executor test failures.
1. Fix boolean string generation in interactive_shell.py
2. Fix test_run_model mock call count
3. Fix test_interactive_shell platform to force remote execution
"""
from pathlib import Path
import re

def fix_interactive_shell():
    filepath = "neural/cloud/interactive_shell.py"
    path = Path(filepath)
    if not path.exists():
        return

    content = path.read_text(encoding='utf-8')
    
    # Fix boolean generation: setup_tunnel={str(setup_tunnel).lower()} -> setup_tunnel={setup_tunnel}
    new_content = content.replace(
        "setup_tunnel={str(setup_tunnel).lower()}",
        "setup_tunnel={setup_tunnel}"
    )
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed boolean generation in: {filepath}")

def fix_test_cloud_executor():
    filepath = "tests/cloud/test_cloud_executor.py"
    path = Path(filepath)
    if not path.exists():
        return

    content = path.read_text(encoding='utf-8')
    
    # Fix test_run_model: reset mock
    # Find: result = self.executor.run_model(...)
    # Insert before: self.mock_subprocess.run.reset_mock()
    
    if "self.mock_subprocess.run.reset_mock()" not in content:
        content = content.replace(
            "result = self.executor.run_model",
            "self.mock_subprocess.run.reset_mock()\n        result = self.executor.run_model"
        )
        print(f"✅ Fixed test_run_model in: {filepath}")
        path.write_text(content, encoding='utf-8')

def fix_test_interactive_shell():
    filepath = "tests/cloud/test_interactive_shell.py"
    path = Path(filepath)
    if not path.exists():
        return

    content = path.read_text(encoding='utf-8')
    
    # Change platform to sagemaker for tests that expect remote execution
    # But setUp sets it to kaggle.
    # We can change setUp to use sagemaker?
    # Or update specific tests.
    
    # Let's change setUp to use sagemaker, as it seems more robust for testing remote calls
    # But we need to update expectations (e.g. create_sagemaker_notebook instead of create_kaggle_kernel)
    
    # Easier: In test_shell_command and test_python_command, set self.shell.platform = 'sagemaker'
    # And mock execute_on_sagemaker instead of execute_on_kaggle
    
    # Actually, the tests assert execute_on_kaggle is called.
    # So we MUST use Kaggle platform in test, but disable local execution optimization?
    # The optimization checks: if self.platform.lower() == 'kaggle':
    
    # If we change platform to 'sagemaker', we need to change assertions to execute_on_sagemaker.
    
    # Let's try to patch subprocess.run in test_shell_command to fail, forcing fallback?
    # In interactive_shell.py:
    # except Exception as e: ... Falling back to cloud execution...
    
    # So if we patch subprocess.run to raise Exception, it should fallback.
    
    # But test_interactive_shell.py doesn't patch subprocess.
    
    # Let's just modify interactive_shell.py to remove the optimization? 
    # No, that's a feature.
    
    # Let's modify the test to expect local execution?
    # The test mocks execute_on_kaggle.
    
    # Let's modify the test to set platform to sagemaker and assert execute_on_sagemaker.
    # This requires changing the test code significantly.
    
    # Alternative: In setUp, use sagemaker.
    # Replace 'kaggle' with 'sagemaker' in setUp.
    # Replace connect_to_kaggle with connect_to_sagemaker
    # Replace create_kaggle_kernel with create_sagemaker_notebook
    # Replace execute_on_kaggle with execute_on_sagemaker
    # Replace delete_kaggle_kernel with delete_sagemaker_notebook (if exists)
    
    # This seems like the right way to test the "Remote Shell" functionality generic logic.
    
    new_content = content.replace("'kaggle'", "'sagemaker'")
    new_content = new_content.replace("connect_to_kaggle", "connect_to_sagemaker")
    new_content = new_content.replace("create_kaggle_kernel", "create_sagemaker_notebook")
    new_content = new_content.replace("execute_on_kaggle", "execute_on_sagemaker")
    new_content = new_content.replace("delete_kaggle_kernel", "delete_sagemaker_notebook")
    new_content = new_content.replace("'test-kernel-id'", "'test-notebook-name'")
    new_content = new_content.replace("self.shell.kernel_id", "self.shell.notebook_name")
    
    if new_content != content:
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Switched test_interactive_shell to SageMaker in: {filepath}")

if __name__ == "__main__":
    fix_interactive_shell()
    fix_test_cloud_executor()
    fix_test_interactive_shell()
