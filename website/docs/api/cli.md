---
sidebar_position: 1
---

# CLI Reference

Complete command-line interface reference for Neural DSL.

## Installation

```bash
pip install neural-dsl[full]
```

## Global Options

```bash
neural [OPTIONS] COMMAND [ARGS]
```

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message |
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress non-error output |

## Commands

### compile

Compile DSL to framework-specific code.

```bash
neural compile [OPTIONS] FILE
```

**Options:**
- `--backend` - Target framework: `tensorflow`, `pytorch`, `onnx` (default: `tensorflow`)
- `--output` - Output file path (default: `generated_<backend>.py`)
- `--optimize` - Enable optimizations

**Examples:**

```bash
# Compile to TensorFlow
neural compile model.neural --backend tensorflow

# Compile to PyTorch with custom output
neural compile model.neural --backend pytorch --output my_model.py

# Compile with optimizations
neural compile model.neural --backend tensorflow --optimize
```

### run

Compile and execute model training.

```bash
neural run [OPTIONS] FILE
```

**Options:**
- `--backend` - Target framework (default: `tensorflow`)
- `--output` - Save generated code
- `--data` - Path to training data
- `--epochs` - Override epochs from DSL
- `--batch-size` - Override batch size

**Examples:**

```bash
# Run with TensorFlow
neural run model.neural --backend tensorflow

# Override training parameters
neural run model.neural --epochs 50 --batch-size 128

# Specify data path
neural run model.neural --data ./datasets/mnist
```

### visualize

Generate architecture visualizations.

```bash
neural visualize [OPTIONS] FILE
```

**Options:**
- `--format` - Output format: `png`, `svg`, `html`, `pdf`
- `--output` - Output directory (default: current)
- `--show` - Open visualization after generation

**Examples:**

```bash
# Generate PNG diagram
neural visualize model.neural --format png

# Generate interactive HTML
neural visualize model.neural --format html --show

# Multiple formats
neural visualize model.neural --format png --format html
```

### debug

Start NeuralDbg debugger.

```bash
neural debug [OPTIONS] FILE
```

**Options:**
- `--port` - Dashboard port (default: 8050)
- `--backend` - Framework to use
- `--gradients` - Focus on gradient analysis
- `--dead-neurons` - Detect dead neurons
- `--anomalies` - Monitor anomalies
- `--step` - Enable step debugging

**Examples:**

```bash
# Basic debugging
neural debug model.neural

# Gradient analysis
neural debug model.neural --gradients

# Custom port
neural debug model.neural --port 8888

# Step debugging
neural debug model.neural --step
```

### export

Export model for deployment.

```bash
neural export [OPTIONS] FILE
```

**Options:**
- `--format` - Export format: `onnx`, `tflite`, `torchscript`, `savedmodel`
- `--optimize` - Enable optimizations
- `--quantize` - Enable quantization
- `--quantization-type` - Quantization type: `int8`, `float16`
- `--output` - Output path
- `--deployment` - Generate deployment config: `tfserving`, `torchserve`

**Examples:**

```bash
# Export to ONNX
neural export model.neural --format onnx --optimize

# Export to TFLite with quantization
neural export model.neural --format tflite --quantize --quantization-type int8

# Export with deployment config
neural export model.neural --format savedmodel --deployment tfserving
```

### docs

Generate documentation from DSL file.

```bash
neural docs [OPTIONS] FILE
```

**Options:**
- `--output` - Output file (default: `model.md`)
- `--format` - Format: `markdown`, `html`, `pdf`
- `--include-shapes` - Include shape information
- `--include-params` - Include parameter counts

**Examples:**

```bash
# Generate Markdown
neural docs model.neural --output README.md

# Generate HTML with shapes
neural docs model.neural --format html --include-shapes

# Generate PDF
neural docs model.neural --format pdf --include-params
```

### validate

Validate DSL file without compilation.

```bash
neural validate [OPTIONS] FILE
```

**Options:**
- `--strict` - Enable strict validation
- `--check-shapes` - Validate shape propagation

**Examples:**

```bash
# Basic validation
neural validate model.neural

# Strict validation with shapes
neural validate model.neural --strict --check-shapes
```

### clean

Remove generated files.

```bash
neural clean [OPTIONS]
```

**Options:**
- `--all` - Remove all generated files
- `--generated` - Remove generated code
- `--visualizations` - Remove visualization files
- `--exports` - Remove exported models
- `--yes` - Skip confirmation

**Examples:**

```bash
# Dry run (default)
neural clean --all

# Actually remove files
neural clean --all --yes

# Remove only generated code
neural clean --generated --yes
```

### track

Experiment tracking commands.

```bash
neural track [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**
- `list` - List all experiments
- `show` - Show experiment details
- `compare` - Compare experiments
- `plot` - Plot metrics

**Examples:**

```bash
# List experiments
neural track list

# Show experiment
neural track show <experiment_id>

# Compare two experiments
neural track compare <exp1> <exp2>

# Plot metrics
neural track plot <experiment_id>
```

### cloud

Cloud integration commands.

```bash
neural cloud [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**
- `connect` - Connect to cloud platform
- `execute` - Execute on cloud
- `run` - Run with tunnel setup

**Examples:**

```bash
# Connect to Kaggle
neural cloud connect kaggle

# Execute on cloud
neural cloud execute kaggle model.neural

# Run with tunnel
neural cloud run --setup-tunnel
```

## Configuration File

Create `neural_config.yaml` for default settings:

```yaml
default_backend: tensorflow
visualization:
  default_format: html
  auto_open: true
debug:
  default_port: 8050
  theme: dark
tracking:
  auto_track: true
  experiment_dir: ./neural_experiments
```

## Environment Variables

- `NEURAL_BACKEND` - Default backend
- `NEURAL_CONFIG` - Config file path
- `NEURAL_CACHE_DIR` - Cache directory
- `NEURAL_LOG_LEVEL` - Logging level

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Invalid DSL syntax
- `3` - Shape mismatch
- `4` - Backend error
- `5` - File not found

## Shell Completion

Enable tab completion:

```bash
# Bash
eval "$(_NEURAL_COMPLETE=bash_source neural)"

# Zsh
eval "$(_NEURAL_COMPLETE=zsh_source neural)"

# Fish
_NEURAL_COMPLETE=fish_source neural | source
```

## See Also

- [Python API](/docs/api/python-api)
- [Tutorial: CLI Usage](/docs/tutorial/cli-usage)
- [Guide: Workflow Optimization](/docs/guides/workflow)
