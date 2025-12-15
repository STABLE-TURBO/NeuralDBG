# Neural DSL - Linting and Code Quality Guide

## Overview

This guide provides instructions for running linting and type checking tools to ensure code quality in the Neural DSL project.

## Prerequisites

Install development dependencies:

```bash
# Option 1: Install from requirements-dev.txt
pip install -r requirements-dev.txt

# Option 2: Install tools directly
pip install ruff mypy isort
```

## Quick Start

### Run All Checks

```bash
# 1. Fix import ordering
python scripts/fix_imports.py

# 2. Run ruff auto-fix
python -m ruff check . --fix

# 3. Format code with ruff
python -m ruff format .

# 4. Run final linting check
python -m ruff check .

# 5. Run type checking
python -m mypy neural/ --ignore-missing-imports
```

### Individual Commands

#### 1. Import Ordering (isort)

Sort imports according to PEP 8:
```bash
python -m isort neural/ tests/ --profile black
```

Or use the custom script:
```bash
python scripts/fix_imports.py
python scripts/fix_imports.py --dry-run  # Preview changes
```

#### 2. Linting with Ruff

Check for linting issues:
```bash
python -m ruff check .
```

Auto-fix issues:
```bash
python -m ruff check . --fix
```

Format code:
```bash
python -m ruff format .
```

#### 3. Type Checking with Mypy

Run type checking on the neural package:
```bash
python -m mypy neural/ --ignore-missing-imports
```

Check specific modules with strict typing:
```bash
python -m mypy neural/code_generation/ --ignore-missing-imports
python -m mypy neural/utils/ --ignore-missing-imports
python -m mypy neural/shape_propagation/ --ignore-missing-imports
```

## Configuration

### Ruff Configuration

Located in `pyproject.toml`:
- Line length: 100 characters
- Target version: Python 3.11
- Enabled rules: pycodestyle (E), pyflakes (F), isort (I)

### Mypy Configuration

Located in `mypy.ini`:
- Python version: 3.11
- Ignore missing imports: True
- Different strictness levels for different modules:
  - **Priority 1** (Strict): code_generation, utils, shape_propagation
  - **Priority 2** (Moderate): parser, cli, dashboard
  - **Priority 3** (Relaxed): hpo, visualization, training

### Import Ordering

According to isort/ruff configuration:
1. **Future imports**: `from __future__ import annotations`
2. **Standard library**: `import os`, `from typing import Dict`
3. **Third-party**: `import numpy`, `import torch`
4. **First-party**: `from neural.parser import ...`
5. **Local**: `from .utils import ...`

Within each group, imports are sorted alphabetically.

## Common Issues and Fixes

### 1. Import Order Issues

**Problem**: Imports are not sorted correctly

**Fix**: Run the import fixer
```bash
python scripts/fix_imports.py
```

**Example**:
```python
# Before
from neural.parser import Parser
import os
from typing import Dict

# After
import os
from typing import Dict

from neural.parser import Parser
```

### 2. Line Too Long (E501)

**Problem**: Lines exceed 100 characters

**Fix**: Break long lines
```python
# Before
def my_function(param1, param2, param3, param4, param5, param6, param7, param8):
    pass

# After
def my_function(
    param1, param2, param3, param4,
    param5, param6, param7, param8
):
    pass
```

### 3. Unused Imports (F401)

**Problem**: Import is not used in the file

**Fix**: Remove unused imports or add to `__all__`
```python
# If truly unused, remove it
# If needed for public API, add to __all__
__all__ = ['imported_function']
```

### 4. Undefined Name (F821)

**Problem**: Variable or function used but not defined

**Fix**: Import or define the name
```python
# Add import
from neural.exceptions import NeuralException

# Or define it
def my_function():
    pass
```

### 5. Type Annotation Issues

**Problem**: Missing or incorrect type annotations

**Fix**: Add proper type hints
```python
# Before
def process_data(data):
    return data

# After
def process_data(data: List[int]) -> List[int]:
    return data
```

### 6. Import from Untyped Module

**Problem**: Mypy warning about importing from untyped module

**Fix**: Add `# type: ignore` or configure mypy to ignore
```python
import some_untyped_module  # type: ignore
```

## CI/CD Integration

The GitHub Actions CI pipeline automatically runs these checks:

```yaml
- name: Lint with Ruff
  run: ruff check .

- name: Type check with Mypy
  run: mypy neural/ --ignore-missing-imports
```

All checks must pass before merging PRs.

## Pre-commit Hooks

Install pre-commit hooks to run checks automatically before commits:

```bash
pre-commit install
```

This will run:
- Ruff linting and formatting
- Mypy type checking
- Import sorting

## Manual Fixes Applied

The following files have been manually fixed for import ordering:

1. `neural/__init__.py` - Sorted standard library and typing imports
2. `neural/parser/parser.py` - Moved local imports after third-party, moved type aliases after imports
3. `neural/code_generation/code_generator.py` - Consolidated and sorted local imports
4. `neural/shape_propagation/shape_propagator.py` - Moved neural.exceptions imports before neural.parser

## Scripts

### `scripts/fix_imports.py`

Automatically fixes import ordering in all Python files:

```bash
# Fix all files
python scripts/fix_imports.py

# Preview changes (dry-run)
python scripts/fix_imports.py --dry-run
```

### `scripts/lint_and_fix.py`

Comprehensive script that:
1. Installs required tools
2. Runs isort
3. Runs ruff check --fix
4. Runs ruff format
5. Runs final linting check
6. Runs mypy type checking

```bash
python scripts/lint_and_fix.py
```

## Best Practices

1. **Run linting before committing**
   ```bash
   python -m ruff check . --fix
   python -m ruff format .
   ```

2. **Fix type errors incrementally**
   - Start with core modules (code_generation, utils, shape_propagation)
   - Add type hints to function signatures
   - Use `Optional` for nullable parameters
   - Use `Union` for multiple types

3. **Keep imports clean**
   - Remove unused imports
   - Group related imports
   - Use absolute imports for neural modules
   - Use relative imports for same-package modules

4. **Follow PEP 8**
   - 100-character line length (project standard)
   - 2 blank lines between top-level definitions
   - 1 blank line between methods
   - No trailing whitespace

5. **Type hints**
   - Use `from __future__ import annotations` for forward references
   - Prefer built-in types over typing module when possible (Python 3.9+)
   - Use `Dict[str, Any]` over bare `dict`
   - Use `List[int]` over bare `list`

## Troubleshooting

### Ruff not installed

```bash
pip install ruff
# or
pip install -r requirements-dev.txt
```

### Mypy not installed

```bash
pip install mypy
# or
pip install -r requirements-dev.txt
```

### Import errors during linting

Some imports may fail during linting if optional dependencies aren't installed. This is expected and can be ignored for modules that gracefully handle missing dependencies.

### Type errors in optional modules

Modules with relaxed mypy settings (hpo, visualization, etc.) may show type errors. These are acceptable for now but should be improved incrementally.

## Summary

To ensure code quality:

1. **Before committing**: Run `python scripts/fix_imports.py` and `ruff check . --fix`
2. **Before pushing**: Ensure `ruff check .` and `mypy neural/ --ignore-missing-imports` pass
3. **During review**: Address any linting or type checking issues raised by CI

For questions or issues, refer to the project's contribution guidelines in `CONTRIBUTING.md`.
