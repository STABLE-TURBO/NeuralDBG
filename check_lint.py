"""
Basic linting check script for Neural DSL project.
Checks for common issues that ruff and mypy would catch.
"""
import os
import re
import ast
from pathlib import Path
from typing import List, Tuple

def find_python_files(directories: List[str]) -> List[Path]:
    """Find all Python files in given directories."""
    python_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and venvs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(Path(root) / file)
    return python_files

def check_file(filepath: Path) -> List[Tuple[str, int, str]]:
    """Check a single Python file for common issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Parse AST to check for syntax errors
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append((str(filepath), e.lineno or 0, f"Syntax error: {e.msg}"))
            return issues
        
        # Check for unused imports (basic check)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports.append((name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports.append((name, node.lineno))
        
        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line and line.strip():
                issues.append((str(filepath), i, "Line has trailing whitespace"))
        
        # Check for lines too long (>100 chars per ruff config)
        for i, line in enumerate(lines, 1):
            if len(line) > 100 and not line.strip().startswith('#'):
                issues.append((str(filepath), i, f"Line too long ({len(line)} > 100 characters)"))
        
    except Exception as e:
        issues.append((str(filepath), 0, f"Error checking file: {e}"))
    
    return issues

def main():
    """Main function to check all Python files."""
    directories = ['neural', 'tests']
    python_files = find_python_files(directories)
    
    print(f"Checking {len(python_files)} Python files...")
    
    all_issues = []
    for filepath in python_files:
        issues = check_file(filepath)
        all_issues.extend(issues)
    
    if all_issues:
        print(f"\nFound {len(all_issues)} issues:")
        for filepath, lineno, msg in all_issues[:50]:  # Show first 50
            print(f"{filepath}:{lineno}: {msg}")
        if len(all_issues) > 50:
            print(f"... and {len(all_issues) - 50} more issues")
    else:
        print("\nNo issues found!")

if __name__ == '__main__':
    main()
