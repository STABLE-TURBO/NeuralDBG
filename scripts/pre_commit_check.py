#!/usr/bin/env python
"""
Pre-commit check script for Neural DSL.

This script runs linting and type checking to ensure code quality before commits.
Can be used as a manual check or integrated with git hooks.

Usage:
    python scripts/pre_commit_check.py          # Run all checks
    python scripts/pre_commit_check.py --fast   # Skip slower checks (type checking)
    python scripts/pre_commit_check.py --fix    # Auto-fix issues when possible
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str, str]:
    """
    Run a command and return success status.
    
    Returns:
        (success, stdout, stderr)
    """
    print_info(f"Running: {description}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        success = result.returncode == 0
        
        if success:
            print_success(description)
        else:
            print_error(f"{description} failed")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        
        return success, result.stdout, result.stderr
        
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        print_warning(f"Install with: pip install {cmd[0]}")
        return False, "", f"Command not found: {cmd[0]}"


def check_tool_available(tool: str) -> bool:
    """Check if a tool is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", tool, "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def main() -> int:
    """
    Main function to run all pre-commit checks.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = sys.argv[1:]
    fast_mode = '--fast' in args
    fix_mode = '--fix' in args
    
    print_header("Neural DSL - Pre-Commit Checks")
    
    if fast_mode:
        print_info("Running in fast mode (skipping type checking)")
    if fix_mode:
        print_info("Auto-fix mode enabled")
    
    # Check which tools are available
    print_header("Checking Available Tools")
    
    tools = {
        'ruff': check_tool_available('ruff'),
        'mypy': check_tool_available('mypy'),
        'isort': check_tool_available('isort'),
    }
    
    for tool, available in tools.items():
        if available:
            print_success(f"{tool} is available")
        else:
            print_warning(f"{tool} is not available")
    
    if not any(tools.values()):
        print_error("No linting tools available!")
        print_info("Install with: pip install -r requirements-dev.txt")
        return 1
    
    checks_passed = []
    checks_failed = []
    
    # Check 1: Import ordering with custom script
    print_header("Check 1: Import Ordering")
    fix_imports_script = Path(__file__).parent / "fix_imports.py"
    if fix_imports_script.exists():
        mode = [] if fix_mode else ["--dry-run"]
        success, stdout, stderr = run_command(
            [sys.executable, str(fix_imports_script)] + mode,
            "Import ordering check"
        )
        if success:
            checks_passed.append("Import ordering")
        else:
            checks_failed.append("Import ordering")
    else:
        print_warning("Import fixer script not found, skipping")
    
    # Check 2: Ruff linting
    if tools['ruff']:
        print_header("Check 2: Ruff Linting")
        
        if fix_mode:
            # Run with --fix
            success, _, _ = run_command(
                [sys.executable, "-m", "ruff", "check", ".", "--fix"],
                "Ruff check (with auto-fix)"
            )
        else:
            # Run without --fix
            success, _, _ = run_command(
                [sys.executable, "-m", "ruff", "check", "."],
                "Ruff check"
            )
        
        if success:
            checks_passed.append("Ruff linting")
        else:
            checks_failed.append("Ruff linting")
            print_warning("Run 'ruff check . --fix' to auto-fix issues")
    
    # Check 3: Ruff formatting
    if tools['ruff']:
        print_header("Check 3: Code Formatting")
        
        if fix_mode:
            success, _, _ = run_command(
                [sys.executable, "-m", "ruff", "format", "."],
                "Ruff format"
            )
        else:
            success, _, _ = run_command(
                [sys.executable, "-m", "ruff", "format", ".", "--check"],
                "Ruff format check"
            )
        
        if success:
            checks_passed.append("Code formatting")
        else:
            checks_failed.append("Code formatting")
            print_warning("Run 'ruff format .' to format code")
    
    # Check 4: Type checking with mypy (skip in fast mode)
    if tools['mypy'] and not fast_mode:
        print_header("Check 4: Type Checking")
        
        success, _, _ = run_command(
            [sys.executable, "-m", "mypy", "neural/", "--ignore-missing-imports"],
            "Mypy type checking"
        )
        
        if success:
            checks_passed.append("Type checking")
        else:
            checks_failed.append("Type checking")
            print_warning("Some type errors found - review mypy output above")
    elif not fast_mode:
        print_warning("Mypy not available, skipping type checking")
    
    # Summary
    print_header("Summary")
    
    if checks_passed:
        print_success(f"Passed checks: {', '.join(checks_passed)}")
    
    if checks_failed:
        print_error(f"Failed checks: {', '.join(checks_failed)}")
        print_info("\nTo fix issues:")
        print("  1. Run: python scripts/pre_commit_check.py --fix")
        print("  2. Review and commit changes")
        print("  3. Re-run this script to verify")
    else:
        print_success("\n🎉 All checks passed! Ready to commit.")
    
    return 0 if not checks_failed else 1


if __name__ == "__main__":
    sys.exit(main())
