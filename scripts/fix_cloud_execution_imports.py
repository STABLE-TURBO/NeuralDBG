"""
Script to fix imports in cloud_execution.py to support mocking.
1. Move visualize_model import to top level
2. Add global ngrok variable and update usage
"""
from pathlib import Path
import re

filepath = "neural/cloud/cloud_execution.py"

def fix_imports():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # 1. Add visualize_model to top level imports
    if "from neural.visualization.visualizer import visualize_model" not in content:
        target = "from neural.cli.cli_aesthetics import print_info, print_success, print_error, print_warning"
        replacement = target + "\n    from neural.visualization.visualizer import visualize_model"
        content = content.replace(target, replacement)
    
    # 2. Remove local import of visualize_model
    # Pattern: from neural.visualization.visualizer import visualize_model
    # But only inside the function (indented)
    content = re.sub(r"^\s+from neural.visualization.visualizer import visualize_model\n", "", content, flags=re.MULTILINE)
    
    # 3. Add ngrok = None at top level
    if "ngrok = None" not in content:
        # Add it after imports
        import_end = content.find("class CloudExecutor:")
        content = content[:import_end] + "ngrok = None\n\n" + content[import_end:]
    
    # 4. Update setup_ngrok_tunnel to use global ngrok
    # We need to replace the body of setup_ngrok_tunnel
    # This is hard with regex. Let's replace the specific block.
    
    old_ngrok_block = """            # Try to import pyngrok
            try:
                from pyngrok import ngrok
            except ImportError:
                print_info("Installing pyngrok...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                from pyngrok import ngrok

            # Start the tunnel
            public_url = ngrok.connect(port).public_url"""
            
    new_ngrok_block = """            global ngrok
            if ngrok is None:
                # Try to import pyngrok
                try:
                    from pyngrok import ngrok as _ngrok
                except ImportError:
                    print_info("Installing pyngrok...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                    from pyngrok import ngrok as _ngrok
                ngrok = _ngrok

            # Start the tunnel
            public_url = ngrok.connect(port).public_url"""
            
    content = content.replace(old_ngrok_block, new_ngrok_block)
    
    # 5. Update cleanup to use global ngrok
    old_cleanup_block = """        # Clean up ngrok tunnels if pyngrok is installed
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except ImportError:
            pass"""
            
    new_cleanup_block = """        # Clean up ngrok tunnels if pyngrok is installed
        global ngrok
        if ngrok:
            try:
                ngrok.kill()
            except Exception:
                pass"""
                
    content = content.replace(old_cleanup_block, new_cleanup_block)
    
    path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed imports in: {filepath}")

if __name__ == "__main__":
    fix_imports()
