# Logging Implementation Guide

## Overview

This document describes the implementation of proper logging using `logging.getLogger(__name__)` across the Neural DSL codebase, replacing all `print()` statements with structured logging.

## Implementation Complete

### ✅ Core Modules Migrated

The following critical modules have been fully migrated:

1. **neural/cli/cli.py** - Main CLI interface
   - Logger initialized at line 98
   - ~25+ print statements converted to appropriate log levels
   - Integrated with verbose flag (`--verbose`)

2. **neural/shape_propagation/shape_propagator.py** - Shape propagation engine
   - Logger initialized at line 20
   - ~12+ DEBUG print statements converted to logger.debug()
   - Step debugging info uses logger.info()

3. **neural/dashboard/dashboard.py** - Real-time debugging dashboard
   - Already using proper logging via get_logger()
   - No migration needed (exemplary implementation)

### 🔧 Logging Configuration

The logging system is configured in `neural/cli/cli.py` via `configure_logging()`:

```python
def configure_logging(verbose: bool = False) -> None:
    """Configure logging levels based on verbosity."""
    # Suppress noisy third-party libraries
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
    os.environ['MPLBACKEND'] = 'Agg'          # Matplotlib

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Configure Neural logger with detailed format for verbose mode
    neural_logger = logging.getLogger('neural')
    neural_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose 
        else "%(levelname)s: %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    neural_logger.handlers = [handler]

    # Silence noisy libraries
    for logger_name in ['graphviz', 'matplotlib', 'tensorflow', 'jax', 
                        'torch', 'optuna', 'dash', 'plotly', ...]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
```

### 📊 Log Level Standards

All modules follow these consistent log level guidelines:

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Detailed diagnostic info, variable dumps, internal state | `logger.debug(f"_handle_conv2d - kernel: {kernel}, stride: {stride}")` |
| **INFO** | General informational messages, progress, results | `logger.info(f"Compilation successful!")` |
| **WARNING** | Potential issues that don't prevent execution | `logger.warning(f"Dataset '{dataset}' may not be supported")` |
| **ERROR** | Failures and exceptions | `logger.error(f"Parsing failed: {str(e)}")` |

### 🎯 Standard Logger Initialization

All Python modules use this pattern:

```python
import logging

logger = logging.getLogger(__name__)
```

This ensures:
- Each module has its own logger with a hierarchical name (e.g., `neural.cli.cli`)
- Logger configuration can be controlled globally via the `neural` logger
- Logs can be filtered by module name

### 📝 Migration Pattern Examples

#### Before: Simple Print
```python
print(f"Processing {file_name}")
```

#### After: Logger Info
```python
logger.info(f"Processing {file_name}")
```

#### Before: Debug Print
```python
print(f"DEBUG: input_shape={input_shape}, params={params}")
```

#### After: Logger Debug
```python
logger.debug(f"input_shape={input_shape}, params={params}")
```

#### Before: Error Print
```python
print(f"ERROR: Failed to load file: {error}")
```

#### After: Logger Error
```python
logger.error(f"Failed to load file: {error}")
```

### 🚀 Usage

#### Normal Mode (INFO and above)
```bash
neural compile model.neural --backend tensorflow
```
Output shows:
- INFO: General progress and results
- WARNING: Potential issues
- ERROR: Failures

#### Verbose Mode (DEBUG and above)
```bash
neural --verbose compile model.neural --backend tensorflow
```
Output shows all of the above plus:
- DEBUG: Detailed diagnostic information
- Variable values
- Internal state transitions

### ⚠️ Exceptions: CLI Aesthetics

The following functions are intentionally NOT migrated as they provide formatted, colored output for the CLI user interface:

- `print_error()` - Red error messages with formatting
- `print_info()` - Blue informational messages
- `print_success()` - Green success messages
- `print_warning()` - Yellow warning messages
- `print_command_header()` - Formatted command headers
- `print_help_command()` - Help text display
- `print_neural_logo()` - ASCII art logo

These functions are in `neural/cli/cli_aesthetics.py` and serve a different purpose than diagnostic logging.

### 🔍 Verification

To verify logging implementation:

1. **Check for remaining print statements:**
   ```bash
   # Find print statements (excluding print_* functions)
   grep -r "^\s*print(" neural/ --include="*.py" | grep -v "print_" | grep -v "code +="
   ```

2. **Test verbose mode:**
   ```bash
   neural --verbose compile examples/mnist.neural
   ```
   Should show DEBUG messages with timestamps

3. **Test normal mode:**
   ```bash
   neural compile examples/mnist.neural
   ```
   Should show only INFO/WARNING/ERROR without timestamps

4. **Verify logger initialization:**
   ```bash
   grep -r "logger = logging.getLogger(__name__)" neural/ --include="*.py"
   ```

### 📦 Tools Provided

Two scripts are provided to assist with migration:

#### 1. `replace_print_with_logging.py`
Basic automated migration script (initial implementation)

#### 2. `complete_logging_migration.py`
Enhanced migration script with:
- Dry-run mode to preview changes
- Single file processing
- Priority-only mode for core modules
- Intelligent log level detection
- Automatic logger initialization

**Usage:**
```bash
# Dry run to see what would change
python complete_logging_migration.py --dry-run

# Process only priority files
python complete_logging_migration.py --priority-only

# Process a specific file
python complete_logging_migration.py --file neural/parser/parser.py

# Process all files
python complete_logging_migration.py
```

### 📋 Remaining Work

The following files still contain print() statements and should be migrated:

**High Priority (15 files):**
- neural/code_generation/tensorflow_generator.py
- neural/code_generation/onnx_generator.py
- neural/hpo/*.py (HPO modules)
- neural/visualization/*.py (Visualization modules)

**Medium Priority (30+ files):**
- tests/performance/*.py (Performance tests)
- tests/integration_tests/*.py (Integration tests)
- scripts/*.py (Automation scripts)
- tools/profiler/*.py (Profiling tools)

**Low Priority (110+ files):**
- examples/*.py (Example scripts - print() may be appropriate for examples)
- tests/**/*.py (Test files - consider if logging is needed)
- neural/aquarium/*.py (IDE components - separate project scope)
- neural/benchmarks/*.py (Benchmarking - print() may be appropriate)

### 🎓 Best Practices

1. **Choose Appropriate Log Levels:**
   - Use DEBUG for detailed diagnostics that help developers debug issues
   - Use INFO for general progress and status updates
   - Use WARNING for recoverable issues or deprecated features
   - Use ERROR for failures and exceptions

2. **Include Context in Log Messages:**
   ```python
   # Good
   logger.debug(f"Processing layer {layer_name} with shape {input_shape}")
   
   # Less helpful
   logger.debug("Processing layer")
   ```

3. **Don't Log Sensitive Information:**
   ```python
   # Bad
   logger.debug(f"User password: {password}")
   
   # Good
   logger.debug(f"User authentication attempt for user {username}")
   ```

4. **Use Exceptions with ERROR Logs:**
   ```python
   try:
       process_model(model_data)
   except Exception as e:
       logger.error(f"Failed to process model: {e}", exc_info=True)
       raise
   ```

5. **Format Messages Consistently:**
   - Start with the action: "Processing...", "Loading...", "Generating..."
   - Include relevant context: file names, layer names, shapes
   - Use f-strings for formatting

### 📖 References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- Neural DSL `AGENTS.md` - Code style guidelines

### 🔄 Future Enhancements

1. **Structured Logging:**
   - Add JSON logging format for production deployments
   - Include correlation IDs for distributed tracing

2. **Log Rotation:**
   - Add file-based logging with rotation
   - Configure log retention policies

3. **Metrics Integration:**
   - Integrate with monitoring systems (Prometheus, Datadog)
   - Add custom metrics for key operations

4. **Performance Logging:**
   - Add timing decorators for critical functions
   - Log performance metrics automatically

5. **Configuration File:**
   - Create `logging.conf` for external configuration
   - Support environment-specific configurations

## Summary

The logging implementation provides:
- ✅ Structured, level-based logging across core modules
- ✅ Verbose mode for detailed debugging
- ✅ Silent third-party libraries
- ✅ Consistent logger initialization pattern
- ✅ Proper log level usage
- ✅ Tools for completing migration
- ✅ Clear documentation and guidelines

The foundation is now in place for professional-grade logging across the entire Neural DSL codebase.
