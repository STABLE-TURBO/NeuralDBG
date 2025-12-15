#!/usr/bin/env python
import subprocess
import sys

# Install the package in editable mode
print("Installing core package...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

print("\nInstalling development dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])

print("\nSetup complete!")
