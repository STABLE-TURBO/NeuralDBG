import subprocess
import sys

# Install dev dependencies
packages = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "pylint>=2.15.0",
    "mypy>=1.0.0",
    "flake8>=6.0.0",
    "pre-commit>=3.0.0",
    "pip-audit>=2.0.0",
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nDevelopment dependencies installed successfully!")
