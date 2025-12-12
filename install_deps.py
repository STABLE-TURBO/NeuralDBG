import subprocess
import sys

# Install the package in editable mode
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
