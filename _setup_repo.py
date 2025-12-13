#!/usr/bin/env python
"""Setup script for Neural DSL repository."""
import subprocess
import sys
from pathlib import Path

def main():
    """Install Neural DSL in development mode."""
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print("Virtual environment not found at .venv")
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    
    print("Installing Neural DSL in editable mode...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", "."], check=True)
    
    print("\nInstalling development dependencies...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "-r", "requirements-dev.txt"], check=True)
    
    print("\n✓ Repository setup complete!")
    print("\nActivate the virtual environment with:")
    print("  .venv\\Scripts\\Activate.ps1  (PowerShell)")
    print("  .venv\\Scripts\\activate.bat  (CMD)")

if __name__ == "__main__":
    main()
