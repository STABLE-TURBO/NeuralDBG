#!/usr/bin/env python3
"""
Script to replace print() statements with proper logging across all Python files.
"""
import os
import re
from pathlib import Path


def needs_logging_import(content: str) -> bool:
    """Check if file needs logging import."""
    return 'import logging' not in content and 'from logging import' not in content


def has_logger_defined(content: str) -> bool:
    """Check if logger is already defined."""
    return bool(re.search(r'logger\s*=\s*logging\.getLogger\(__name__\)', content))


def add_logging_import(content: str) -> str:
    """Add logging import and logger definition if not present."""
    lines = content.split('\n')
    
    # Find the right place to insert imports (after docstring and before first code)
    import_index = 0
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(docstring_char) >= 2:
                    in_docstring = False
                    import_index = i + 1
            elif stripped.endswith(docstring_char):
                in_docstring = False
                import_index = i + 1
            continue
        
        # Skip shebang and encoding
        if stripped.startswith('#!') or 'coding:' in stripped or 'coding=' in stripped:
            import_index = i + 1
            continue
        
        # Found first import or code
        if stripped and not stripped.startswith('#'):
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Find last import
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].strip().startswith('import ') and not lines[j].strip().startswith('from ') and not lines[j].strip().startswith('#'):
                        import_index = j
                        break
            else:
                import_index = i
            break
    
    # Insert logging import if needed
    if needs_logging_import(content):
        lines.insert(import_index, 'import logging')
        lines.insert(import_index + 1, '')
        import_index += 2
    
    # Add logger definition if needed
    if not has_logger_defined(content):
        # Find where to add logger (after imports, before first class/function)
        logger_index = import_index
        for i in range(import_index, len(lines)):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('import ') and not stripped.startswith('from '):
                logger_index = i
                break
        
        lines.insert(logger_index, 'logger = logging.getLogger(__name__)')
        lines.insert(logger_index + 1, '')
    
    return '\n'.join(lines)


def replace_print_statements(content: str, file_path: str) -> tuple[str, int]:
    """Replace print() statements with appropriate logging calls."""
    count = 0
    
    # Patterns to identify print statements (excluding those in comments)
    # Match print statements but not print_* function calls (like print_error, print_info, etc.)
    print_pattern = r'(\s*)print\((.*?)\)(?:\s*#.*)?$'
    
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        # Skip comment lines
        if line.strip().startswith('#'):
            modified_lines.append(line)
            continue
        
        # Check if line contains a print_* function call (e.g., print_error, print_info)
        if re.search(r'print_\w+\(', line):
            modified_lines.append(line)
            continue
        
        # Match print() statements
        match = re.search(print_pattern, line)
        if match:
            indent = match.group(1)
            content_part = match.group(2)
            
            # Determine log level based on content
            lower_content = content_part.lower()
            if 'error' in lower_content or 'fail' in lower_content or 'exception' in lower_content:
                log_level = 'error'
            elif 'warn' in lower_content:
                log_level = 'warning'
            elif 'debug' in lower_content or file_path.endswith('test_*.py') or 'tests/' in file_path:
                log_level = 'debug'
            else:
                log_level = 'info'
            
            # Replace print with logger call
            new_line = f'{indent}logger.{log_level}({content_part})'
            modified_lines.append(new_line)
            count += 1
        else:
            modified_lines.append(line)
    
    return '\n'.join(modified_lines), count


def process_file(file_path: Path) -> tuple[bool, int]:
    """Process a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has print statements (excluding print_* functions)
        if not re.search(r'[^_]print\(', content):
            return False, 0
        
        # Add logging import and logger if needed
        modified_content = add_logging_import(content)
        
        # Replace print statements
        final_content, count = replace_print_statements(modified_content, str(file_path))
        
        if count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            return True, count
        
        return False, 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    """Main function to process all Python files."""
    # Get all Python files
    python_files = []
    
    # Core modules (priority)
    priority_files = [
        'neural/cli/cli.py',
        'neural/parser/parser.py',
        'neural/shape_propagation/shape_propagator.py',
        'neural/dashboard/dashboard.py',
        'neural/code_generation/code_generator.py',
        'neural/code_generation/pytorch_generator.py',
    ]
    
    # Add priority files if they exist
    for file_path in priority_files:
        if Path(file_path).exists():
            python_files.append(Path(file_path))
    
    # Find all other Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        skip_dirs = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', 
                     '.ruff_cache', 'node_modules', 'build', 'dist', '.eggs'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if file_path not in python_files:
                    python_files.append(file_path)
    
    print(f"Processing {len(python_files)} Python files...")
    
    total_replacements = 0
    files_modified = 0
    
    for file_path in python_files:
        modified, count = process_file(file_path)
        if modified:
            files_modified += 1
            total_replacements += count
            print(f"✓ {file_path}: {count} print() statements replaced")
    
    print(f"\nSummary:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total replacements: {total_replacements}")


if __name__ == '__main__':
    main()
