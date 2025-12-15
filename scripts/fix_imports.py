#!/usr/bin/env python
"""
Script to fix import ordering in Python files according to PEP 8 and isort conventions.

Import order should be:
1. Future imports (from __future__ import ...)
2. Standard library imports
3. Third-party imports
4. First-party/local imports

Within each group, imports should be sorted alphabetically.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Standard library modules (common ones)
STDLIB_MODULES = {
    '__future__', 'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'binascii',
    'bisect', 'builtins', 'bz2', 'calendar', 'cmath', 'cmd', 'code', 'codecs',
    'collections', 'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
    'contextvars', 'copy', 'copyreg', 'csv', 'ctypes', 'curses', 'dataclasses',
    'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
    'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
    'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
    'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox',
    'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing',
    'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib',
    'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
    'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib', 'resource',
    'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex',
    'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver',
    'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
    'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time',
    'timeit', 'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 'traceback', 'tracemalloc',
    'tty', 'turtle', 'turtledemo', 'types', 'typing', 'typing_extensions', 'unicodedata',
    'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib',
}

# First-party modules for this project
FIRST_PARTY_MODULES = {'neural', 'pretrained_models'}


def classify_import(import_line: str) -> str:
    """
    Classify an import line as 'future', 'stdlib', 'thirdparty', or 'firstparty'.
    """
    # Future imports
    if import_line.startswith('from __future__'):
        return 'future'
    
    # Extract module name
    if import_line.startswith('from '):
        # from X import Y
        match = re.match(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', import_line)
        if match:
            module = match.group(1)
        else:
            return 'unknown'
    elif import_line.startswith('import '):
        # import X
        match = re.match(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', import_line)
        if match:
            module = match.group(1)
        else:
            return 'unknown'
    else:
        return 'unknown'
    
    # Check classifications
    if module in STDLIB_MODULES:
        return 'stdlib'
    elif module in FIRST_PARTY_MODULES or module.startswith('.'):
        return 'firstparty'
    else:
        return 'thirdparty'


def extract_imports(content: str) -> Tuple[List[str], List[str], int, int]:
    """
    Extract import statements from file content.
    
    Returns:
        - List of import lines
        - List of non-import lines before imports
        - Start line number of imports block
        - End line number of imports block
    """
    lines = content.split('\n')
    imports = []
    pre_import_lines = []
    import_start = -1
    import_end = -1
    in_import_block = False
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                in_docstring = True
                pre_import_lines.append(line)
                if stripped.count(docstring_char) >= 2:  # Single-line docstring
                    in_docstring = False
                continue
        else:
            pre_import_lines.append(line)
            if docstring_char in line:
                in_docstring = False
            continue
        
        # Skip blank lines and comments before imports
        if not stripped or stripped.startswith('#'):
            if not in_import_block:
                pre_import_lines.append(line)
            elif in_import_block and not stripped:
                # Blank line within import block - continue
                pass
            else:
                # Comment or blank after imports started - might be end
                if in_import_block and stripped.startswith('#'):
                    # This could be a comment between imports, include it
                    pass
                elif in_import_block:
                    import_end = i
                    break
            continue
        
        # Check if this is an import line
        if stripped.startswith('import ') or stripped.startswith('from '):
            if not in_import_block:
                in_import_block = True
                import_start = i
            imports.append(line)
        else:
            # Non-import line after imports started
            if in_import_block:
                import_end = i
                break
            else:
                pre_import_lines.append(line)
    
    if import_start != -1 and import_end == -1:
        import_end = import_start + len(imports)
    
    return imports, pre_import_lines, import_start, import_end


def sort_imports(import_lines: List[str]) -> str:
    """
    Sort import lines according to PEP 8 / isort conventions.
    """
    # Classify imports
    future_imports = []
    stdlib_imports = []
    thirdparty_imports = []
    firstparty_imports = []
    unknown_imports = []
    
    for line in import_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        
        classification = classify_import(stripped)
        
        if classification == 'future':
            future_imports.append(stripped)
        elif classification == 'stdlib':
            stdlib_imports.append(stripped)
        elif classification == 'thirdparty':
            thirdparty_imports.append(stripped)
        elif classification == 'firstparty':
            firstparty_imports.append(stripped)
        else:
            unknown_imports.append(stripped)
    
    # Sort each group
    future_imports.sort()
    stdlib_imports.sort()
    thirdparty_imports.sort()
    firstparty_imports.sort()
    
    # Combine groups with blank lines between
    result = []
    
    if future_imports:
        result.extend(future_imports)
        result.append('')
    
    if stdlib_imports:
        result.extend(stdlib_imports)
        result.append('')
    
    if thirdparty_imports:
        result.extend(thirdparty_imports)
        result.append('')
    
    if firstparty_imports:
        result.extend(firstparty_imports)
        result.append('')
    
    if unknown_imports:
        result.extend(unknown_imports)
        result.append('')
    
    # Remove trailing blank line
    while result and result[-1] == '':
        result.pop()
    
    return '\n'.join(result)


def fix_imports_in_file(filepath: Path, dry_run: bool = False) -> bool:
    """
    Fix import ordering in a Python file.
    
    Returns True if changes were made.
    """
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    imports, pre_lines, start, end = extract_imports(content)
    
    if not imports:
        return False
    
    sorted_imports = sort_imports(imports)
    
    # Reconstruct file
    lines = content.split('\n')
    new_content_parts = []
    
    # Add pre-import lines
    new_content_parts.append('\n'.join(pre_lines))
    
    # Add blank line between header and imports if needed
    if pre_lines and pre_lines[-1].strip():
        new_content_parts.append('')
    
    # Add sorted imports
    new_content_parts.append(sorted_imports)
    
    # Add rest of file
    if end < len(lines):
        # Add blank lines after imports
        new_content_parts.append('')
        new_content_parts.append('')
        new_content_parts.append('\n'.join(lines[end:]))
    
    new_content = '\n'.join(new_content_parts)
    
    # Check if content changed
    if new_content.strip() == content.strip():
        return False
    
    if not dry_run:
        try:
            filepath.write_text(new_content, encoding='utf-8')
            print(f"✓ Fixed imports in {filepath}")
        except Exception as e:
            print(f"✗ Error writing {filepath}: {e}")
            return False
    else:
        print(f"Would fix imports in {filepath}")
    
    return True


def main():
    """Main function to process all Python files."""
    repo_root = Path(__file__).parent.parent
    neural_dir = repo_root / 'neural'
    
    if not neural_dir.exists():
        print(f"Error: {neural_dir} not found")
        sys.exit(1)
    
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print("Running in dry-run mode (no changes will be made)\n")
    
    # Process all Python files in neural/
    python_files = list(neural_dir.rglob('*.py'))
    
    # Filter out __pycache__ and build directories
    python_files = [
        f for f in python_files
        if '__pycache__' not in str(f) and 'build' not in str(f)
    ]
    
    print(f"Found {len(python_files)} Python files to check\n")
    
    fixed_count = 0
    for filepath in python_files:
        if fix_imports_in_file(filepath, dry_run):
            fixed_count += 1
    
    print(f"\n{'Would fix' if dry_run else 'Fixed'} imports in {fixed_count} files")
    
    if not dry_run and fixed_count > 0:
        print("\nImport ordering has been fixed. Review changes with: git diff")


if __name__ == '__main__':
    main()
