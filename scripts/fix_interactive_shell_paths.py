"""
Script to fix shlex.split usage in interactive_shell.py for Windows paths.
"""
from pathlib import Path

filepath = "neural/cloud/interactive_shell.py"

def fix_shlex_split():
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️ File not found: {filepath}")
        return

    content = path.read_text(encoding='utf-8')
    
    # Replace shlex.split(arg) with shlex.split(arg, posix=(os.name != 'nt'))
    # We need to make sure we don't double replace if run multiple times
    
    target = "args = shlex.split(arg)"
    replacement = "args = shlex.split(arg, posix=(os.name != 'nt'))"
    
    if target in content:
        new_content = content.replace(target, replacement)
        path.write_text(new_content, encoding='utf-8')
        print(f"✅ Fixed shlex.split usage in: {filepath}")
    else:
        print(f"ℹ️ No changes needed (or target not found) in: {filepath}")

if __name__ == "__main__":
    fix_shlex_split()
