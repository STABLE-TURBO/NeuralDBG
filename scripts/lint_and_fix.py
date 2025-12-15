#!/usr/bin/env python
"""
Comprehensive linting and fixing script for Neural DSL.
Installs necessary tools and runs linting checks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, check: bool = True) -> tuple[int, str, str]:
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"✗ {description} failed with exit code {result.returncode}")
    elif result.returncode == 0:
        print(f"✓ {description} completed successfully")
    
    return result.returncode, result.stdout, result.stderr


def main():
    """Main linting and fixing workflow."""
    print("\n" + "="*60)
    print("Neural DSL - Comprehensive Linting & Fixing")
    print("="*60)
    
    # Step 1: Ensure dev dependencies are installed
    print("\n[1/6] Installing development dependencies...")
    returncode, _, _ = run_command(
        [sys.executable, "-m", "pip", "install", "-q", "ruff", "mypy", "isort"],
        "Installing ruff, mypy, and isort",
        check=False
    )
    
    if returncode != 0:
        print("Warning: Could not install some tools. Continuing with available tools...")
    
    # Step 2: Check which tools are available
    available_tools = {}
    for tool in ["ruff", "mypy", "isort"]:
        result = subprocess.run(
            [sys.executable, "-m", tool, "--version"],
            capture_output=True,
            text=True
        )
        available_tools[tool] = result.returncode == 0
        if available_tools[tool]:
            print(f"✓ {tool} is available")
        else:
            print(f"✗ {tool} is not available")
    
    # Step 3: Run isort to fix import ordering
    if available_tools.get("isort"):
        print("\n[2/6] Running isort to fix import ordering...")
        run_command(
            [sys.executable, "-m", "isort", "neural/", "tests/", "--profile", "black"],
            "Running isort",
            check=False
        )
    else:
        print("\n[2/6] Skipping isort (not available)")
    
    # Step 4: Run ruff check with auto-fix
    if available_tools.get("ruff"):
        print("\n[3/6] Running ruff check with auto-fix...")
        run_command(
            [sys.executable, "-m", "ruff", "check", ".", "--fix"],
            "Running ruff check --fix",
            check=False
        )
    else:
        print("\n[3/6] Skipping ruff (not available)")
    
    # Step 5: Run ruff format
    if available_tools.get("ruff"):
        print("\n[4/6] Running ruff format...")
        run_command(
            [sys.executable, "-m", "ruff", "format", "."],
            "Running ruff format",
            check=False
        )
    else:
        print("\n[4/6] Skipping ruff format (not available)")
    
    # Step 6: Run final linting check
    if available_tools.get("ruff"):
        print("\n[5/6] Running final ruff lint check...")
        returncode, stdout, stderr = run_command(
            [sys.executable, "-m", "ruff", "check", "."],
            "Running ruff check (final)",
            check=False
        )
        
        if returncode == 0:
            print("✓ All ruff checks passed!")
        else:
            print("✗ Some ruff issues remain. Review output above.")
    else:
        print("\n[5/6] Skipping final ruff check (not available)")
    
    # Step 7: Run mypy type checking
    if available_tools.get("mypy"):
        print("\n[6/6] Running mypy type checking...")
        returncode, stdout, stderr = run_command(
            [sys.executable, "-m", "mypy", "neural/", "--ignore-missing-imports"],
            "Running mypy",
            check=False
        )
        
        if returncode == 0:
            print("✓ All mypy checks passed!")
        else:
            print("✗ Some mypy issues found. Review output above.")
    else:
        print("\n[6/6] Skipping mypy (not available)")
    
    print("\n" + "="*60)
    print("Linting and fixing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
