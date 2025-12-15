#!/usr/bin/env python3
"""
Complete logging migration script for Neural DSL project.
This script replaces all print() statements with proper logging calls.

Usage:
    python complete_logging_migration.py [--dry-run] [--file FILE]
    
Options:
    --dry-run    Show what would be changed without making changes
    --file FILE  Process only the specified file
"""

import re
import sys
from pathlib import Path
from typing import Tuple, List


def has_logging_import(lines: List[str]) -> bool:
    """Check if file has logging import."""
    return any('import logging' in line for line in lines)


def has_logger_defined(lines: List[str]) -> bool:
    """Check if logger is defined."""
    return any(re.match(r'\s*logger\s*=\s*logging\.getLogger\(__name__\)', line) for line in lines)


def add_logging_setup(lines: List[str]) -> List[str]:
    """Add logging import and logger definition."""
    result = []
    added_import = False
    added_logger = False
    
    # Find insertion points
    import_idx = 0
    logger_idx = 0
    
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle docstrings
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            docstring_char = '"""' if stripped.startswith('"""') else "'''"
            in_docstring = True
            if stripped.count(docstring_char) >= 2:
                in_docstring = False
                import_idx = i + 1
            result.append(line)
            continue
        elif in_docstring and stripped.endswith(docstring_char):
            in_docstring = False
            import_idx = i + 1
            result.append(line)
            continue
        elif in_docstring:
            result.append(line)
            continue
        
        # Skip shebang and encoding
        if stripped.startswith('#!') or 'coding:' in stripped or 'coding=' in stripped:
            result.append(line)
            import_idx = i + 1
            continue
        
        # Add logging import after other imports
        if not added_import and stripped and not stripped.startswith('#'):
            if stripped.startswith('import ') or stripped.startswith('from '):
                result.append(line)
                # Check if this is the last import
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    if not (next_stripped.startswith('import ') or next_stripped.startswith('from ') or not next_stripped or next_stripped.startswith('#')):
                        if not has_logging_import(lines):
                            result.append('import logging')
                            result.append('')
                        added_import = True
                        logger_idx = len(result)
                continue
            elif not added_import and not has_logging_import(lines):
                result.append('import logging')
                result.append('')
                added_import = True
                logger_idx = len(result)
        
        # Add logger definition before first function/class
        if not added_logger and (stripped.startswith('def ') or stripped.startswith('class ')):
            if not has_logger_defined(lines):
                result.append('logger = logging.getLogger(__name__)')
                result.append('')
            added_logger = True
        
        result.append(line)
    
    # If we haven't added them by the end, add them now
    if not added_import and not has_logging_import(lines):
        result.insert(import_idx, '')
        result.insert(import_idx, 'import logging')
    
    if not added_logger and not has_logger_defined(lines):
        result.insert(logger_idx, '')
        result.insert(logger_idx, 'logger = logging.getLogger(__name__)')
    
    return result


def determine_log_level(content: str, context: str = '') -> str:
    """Determine appropriate log level based on content."""
    lower_content = content.lower()
    lower_context = context.lower()
    
    # Check for explicit level indicators
    if 'debug:' in lower_content or 'debug ' in lower_content:
        return 'debug'
    elif 'error' in lower_content or 'fail' in lower_content or 'exception' in lower_content:
        return 'error'
    elif 'warn' in lower_content or 'warning' in lower_content:
        return 'warning'
    elif 'test' in lower_context or 'example' in lower_context:
        return 'debug'
    else:
        return 'info'


def replace_print_in_line(line: str, line_no: int, file_path: str) -> Tuple[str, bool]:
    """Replace print statement in a single line."""
    # Skip if it's a comment
    if line.strip().startswith('#'):
        return line, False
    
    # Skip print_* function calls (CLI aesthetics)
    if re.search(r'print_\w+\(', line):
        return line, False
    
    # Skip prints in generated code strings (code += "print(...)")
    if re.search(r'code\s*\+=.*print\(', line):
        return line, False
    
    # Match print() statement
    match = re.match(r'(\s*)print\((.*)\)(\s*#.*)?$', line)
    if not match:
        return line, False
    
    indent = match.group(1)
    content = match.group(2)
    comment = match.group(3) or ''
    
    # Determine log level
    log_level = determine_log_level(content, file_path)
    
    # Create new line
    new_line = f'{indent}logger.{log_level}({content}){comment}'
    
    return new_line, True


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """Process a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Add logging setup if needed
        if not has_logging_import(lines) or not has_logger_defined(lines):
            lines = add_logging_setup(lines)
        
        # Replace print statements
        modified_lines = []
        replacements = 0
        
        for i, line in enumerate(lines):
            new_line, replaced = replace_print_in_line(line, i + 1, str(file_path))
            modified_lines.append(new_line)
            if replaced:
                replacements += 1
                if dry_run:
                    print(f"  Line {i+1}: {line.strip()}")
                    print(f"       -> : {new_line.strip()}")
        
        if replacements > 0 and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))
            return True, replacements
        
        return replacements > 0, replacements
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False, 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate print statements to logging')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying them')
    parser.add_argument('--file', type=str, help='Process only specified file')
    parser.add_argument('--priority-only', action='store_true', help='Process only priority files')
    
    args = parser.parse_args()
    
    # Priority files (core functionality)
    priority_files = [
        'neural/cli/cli.py',
        'neural/parser/parser.py',
        'neural/shape_propagation/shape_propagator.py',
        'neural/dashboard/dashboard.py',
        'neural/code_generation/code_generator.py',
        'neural/code_generation/pytorch_generator.py',
    ]
    
    if args.file:
        files_to_process = [Path(args.file)]
    elif args.priority_only:
        files_to_process = [Path(f) for f in priority_files if Path(f).exists()]
    else:
        # Find all Python files
        files_to_process = []
        skip_dirs = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 
                     '.mypy_cache', '.ruff_cache', 'node_modules', 'build', 'dist'}
        
        for path in Path('.').rglob('*.py'):
            if not any(skip_dir in path.parts for skip_dir in skip_dirs):
                files_to_process.append(path)
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files_to_process)} files...")
    print()
    
    total_replacements = 0
    files_modified = 0
    
    for file_path in files_to_process:
        modified, count = process_file(file_path, args.dry_run)
        if modified:
            files_modified += 1
            total_replacements += count
            status = 'WOULD MODIFY' if args.dry_run else 'MODIFIED'
            print(f"✓ {status} {file_path}: {count} print() statements")
            print()
    
    print(f"\nSummary:")
    print(f"  Files processed: {len(files_to_process)}")
    print(f"  Files {'that would be ' if args.dry_run else ''}modified: {files_modified}")
    print(f"  Total replacements: {total_replacements}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
