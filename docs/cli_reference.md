# Neural CLI Reference

Complete reference for all Neural DSL command-line interface commands.

## Table of Contents

- [Global Options](#global-options)
- [Commands](#commands)
  - [compile](#compile)
  - [run](#run)
  - [visualize](#visualize)
  - [debug](#debug)
  - [export](#export)
  - [track](#track)
  - [cloud](#cloud)
  - [clean](#clean)
  - [docs](#docs)
  - [explain](#explain)
  - [version](#version)
  - [help](#help)

## Global Options

Available for all commands:

```bash
--verbose, -v       Enable verbose logging
--cpu               Force CPU mode (disable GPU)
--no-animations     Disable spinners and progress animations
--version           Show version and exit
--help, -h          Show help message and exit
```

Examples:
```bash
neural --version
neural --verbose compile model.neural
neural --cpu --no-animations run model.neural
```

## Commands

### compile

Compile a `.neural` file to executable Python code.

**Syntax:**
```bash
neural compile <file> [OPTIONS]
```

**Options:**
- `--backend, -b` - Target backend: `tensorflow`, `pytorch`, `onnx` (default: `tensorflow`)
- `--dataset` - Dataset name (default: `MNIST`)
- `--output, -o` - Output file path (default: `<file>_<backend>.py`)
- `--dry-run` - Preview generated code without writing to file
- `--hpo` - Enable hyperparameter optimization
- `--auto-flatten-output` - Auto-insert Flatten before Dense/Output when needed

**Examples:**
```bash
# Basic compilation
neural compile model.neural

# Specify backend and output
neural compile model.neural --backend pytorch --output model.py

# Preview without saving
neural compile model.neural --dry-run

# With hyperparameter optimization
neural compile model.neural --hpo --backend tensorflow

# Multiple backends
neural compile model.neural --backend tensorflow --output model_tf.py
neural compile model.neural --backend pytorch --output model_pt.py
```

**Output:**
- Generated Python code with model definition
- Training loop (if specified)
- Data loading utilities

---

### run

Execute a compiled model or optimize and run a `.neural` file.

**Syntax:**
```bash
neural run <file> [OPTIONS]
```

**Options:**
- `--backend, -b` - Backend to use: `tensorflow`, `pytorch` (default: `tensorflow`)
- `--dataset` - Dataset name (default: `MNIST`)
- `--hpo` - Enable HPO for `.neural` files
- `--device, -d` - Device: `auto`, `cpu`, `gpu` (default: `auto`)

**Examples:**
```bash
# Run a Python file
neural run model_tf.py

# Compile and run with HPO
neural run model.neural --hpo --backend tensorflow

# Force CPU execution
neural run model.neural --device cpu

# Use specific dataset
neural run model.neural --dataset CIFAR10
```

---

### visualize

Generate visual representations of network architecture.

**Syntax:**
```bash
neural visualize <file> [OPTIONS]
```

**Options:**
- `--format, -f` - Output format: `html`, `png`, `svg` (default: `html`)
- `--cache/--no-cache` - Use cached visualizations (default: enabled)
- `--attention` - Visualize attention weights (for transformer models)
- `--backend, -b` - Backend for attention visualization
- `--data, -d` - Input data file for attention (`.npy` format)
- `--tokens` - Comma-separated token labels
- `--layer` - Specific attention layer to visualize
- `--head` - Specific attention head to visualize

**Examples:**
```bash
# Generate HTML visualization
neural visualize model.neural

# Generate PNG diagram
neural visualize model.neural --format png

# Visualize attention weights
neural visualize transformer.neural --attention --data input.npy

# Specific attention head
neural visualize transformer.neural --attention --layer encoder_0 --head 2
```

**Output:**
- `architecture.svg` - Network diagram
- `shape_propagation.html` - Interactive shape analysis
- `tensor_flow.html` - Data flow animation
- `attention_outputs/` - Attention visualizations (if `--attention`)

---

### debug

Debug a neural network model with NeuralDbg.

**Syntax:**
```bash
neural debug <file> [OPTIONS]
```

**Options:**
- `--gradients` - Analyze gradient flow
- `--dead-neurons` - Detect dead neurons
- `--anomalies` - Detect training anomalies
- `--step` - Enable step debugging mode
- `--backend, -b` - Backend: `tensorflow`, `pytorch` (default: `tensorflow`)
- `--dataset` - Dataset name (default: `MNIST`)
- `--dashboard, -d` - Start NeuralDbg dashboard
- `--port` - Dashboard port (default: `8050`)

**Examples:**
```bash
# Basic debugging with console output
neural debug model.neural --gradients --dead-neurons

# Launch interactive dashboard
neural debug model.neural --dashboard

# Step-by-step debugging
neural debug model.neural --step

# Full analysis with dashboard
neural debug model.neural --gradients --anomalies --dashboard --port 8080
```

**Dashboard Features:**
- Real-time execution traces
- Gradient flow visualization
- Dead neuron detection
- Memory and FLOPs profiling
- Anomaly detection

---

### export

Export models for deployment.

**Syntax:**
```bash
neural export <file> [OPTIONS]
```

**Options:**
- `--format` - Export format: `onnx`, `tflite`, `torchscript`, `savedmodel`
- `--backend, -b` - Source backend: `tensorflow`, `pytorch`
- `--optimize` - Apply optimization passes
- `--quantize` - Enable quantization
- `--quantization-type` - Type: `int8`, `float16`
- `--deployment` - Deployment target: `tfserving`, `torchserve`, `mobile`

**Examples:**
```bash
# Export to ONNX
neural export model.neural --format onnx --optimize

# TensorFlow Lite with quantization
neural export model.neural --backend tensorflow --format tflite \
  --quantize --quantization-type int8

# TorchScript
neural export model.neural --backend pytorch --format torchscript

# TensorFlow Serving
neural export model.neural --format savedmodel --deployment tfserving
```

---

### track

Experiment tracking commands.

#### track init

Initialize experiment tracking.

**Syntax:**
```bash
neural track init [experiment_name] [OPTIONS]
```

**Options:**
- `--base-dir` - Base directory for experiments (default: `neural_experiments`)
- `--integration` - External tool: `mlflow`, `wandb`, `tensorboard`
- `--project-name` - Project name for W&B
- `--tracking-uri` - MLflow tracking URI
- `--log-dir` - TensorBoard log directory

**Examples:**
```bash
# Simple initialization
neural track init my_experiment

# With MLflow integration
neural track init my_experiment --integration mlflow --tracking-uri http://localhost:5000

# With Weights & Biases
neural track init my_experiment --integration wandb --project-name neural-experiments
```

#### track log

Log data to an experiment.

**Syntax:**
```bash
neural track log [OPTIONS]
```

**Options:**
- `--experiment-id` - Experiment ID (defaults to current)
- `--hyperparameters, -p` - Hyperparameters as JSON
- `--hyperparameters-file, -f` - JSON file with hyperparameters
- `--metrics, -m` - Metrics as JSON
- `--metrics-file` - JSON file with metrics
- `--step` - Step or epoch number
- `--artifact` - Path to artifact file
- `--artifact-name` - Name for the artifact
- `--model` - Path to model file
- `--framework` - Framework used

**Examples:**
```bash
# Log hyperparameters
neural track log -p '{"learning_rate": 0.001, "batch_size": 32}'

# Log metrics
neural track log -m '{"accuracy": 0.95, "loss": 0.05}' --step 10

# Log artifact
neural track log --artifact model.h5 --framework tensorflow

# From file
neural track log --hyperparameters-file params.json
```

#### track list

List all experiments.

**Syntax:**
```bash
neural track list [OPTIONS]
```

**Options:**
- `--base-dir` - Base directory (default: `neural_experiments`)
- `--format, -f` - Output format: `table`, `json` (default: `table`)

**Examples:**
```bash
neural track list
neural track list --format json
```

#### track show

Show experiment details.

**Syntax:**
```bash
neural track show <experiment_id> [OPTIONS]
```

**Options:**
- `--base-dir` - Base directory
- `--format, -f` - Output format: `table`, `json`

**Examples:**
```bash
neural track show exp_123
neural track show exp_123 --format json
```

#### track plot

Plot experiment metrics.

**Syntax:**
```bash
neural track plot <experiment_id> [OPTIONS]
```

**Options:**
- `--base-dir` - Base directory
- `--metrics, -m` - Specific metrics to plot (plots all if not specified)
- `--output, -o` - Output file path (default: `metrics.png`)

**Examples:**
```bash
neural track plot exp_123
neural track plot exp_123 --metrics accuracy loss --output plots.png
```

#### track compare

Compare multiple experiments.

**Syntax:**
```bash
neural track compare <experiment_ids...> [OPTIONS]
```

**Options:**
- `--base-dir` - Base directory
- `--metrics, -m` - Metrics to compare
- `--output-dir, -o` - Output directory (default: `comparison_plots`)

**Examples:**
```bash
neural track compare exp_123 exp_456 exp_789
neural track compare exp_123 exp_456 --metrics accuracy --output-dir results/
```

---

### cloud

Cloud integration commands.

#### cloud run

Run Neural in cloud environments.

**Syntax:**
```bash
neural cloud run [OPTIONS]
```

**Options:**
- `--setup-tunnel` - Set up ngrok tunnel for remote access
- `--port` - Port for no-code interface (default: `8051`)

**Examples:**
```bash
neural cloud run
neural cloud run --setup-tunnel --port 8080
```

#### cloud connect

Connect to a cloud platform.

**Syntax:**
```bash
neural cloud connect <platform> [OPTIONS]
```

**Platforms:**
- `kaggle` - Kaggle notebooks
- `colab` - Google Colab
- `sagemaker` - AWS SageMaker

**Options:**
- `--interactive, -i` - Start interactive shell
- `--notebook, -n` - Start Jupyter-like notebook interface
- `--port` - Notebook server port (default: `8888`)
- `--quiet, -q` - Reduce output verbosity

**Examples:**
```bash
# Connect to Kaggle
neural cloud connect kaggle

# Interactive shell
neural cloud connect colab --interactive

# Notebook interface
neural cloud connect sagemaker --notebook --port 8888
```

#### cloud execute

Execute a Neural DSL file on a cloud platform.

**Syntax:**
```bash
neural cloud execute <platform> <file> [OPTIONS]
```

**Options:**
- `--name` - Name for kernel/notebook

**Examples:**
```bash
neural cloud execute kaggle model.neural
neural cloud execute colab model.neural --name my-experiment
```

---

### clean

Remove generated artifacts.

**Syntax:**
```bash
neural clean [OPTIONS]
```

**Options:**
- `--yes` - Apply deletions (otherwise dry-run)
- `--all` - Also remove caches and artifact directories

**Examples:**
```bash
# Dry run (preview)
neural clean

# Actually delete files
neural clean --yes

# Delete including caches
neural clean --yes --all
```

**Targets:**
- `*_tensorflow.py`, `*_pytorch.py`, `*_onnx.py`
- `architecture.svg`, `architecture.png`
- `shape_propagation.html`, `tensor_flow.html`
- `.neural_cache/`, `comparison_plots/` (with `--all`)

---

### docs

Generate documentation for models.

**Syntax:**
```bash
neural docs <file> [OPTIONS]
```

**Options:**
- `--output, -o` - Output Markdown file (default: `model.md`)
- `--pdf` - Also export to PDF via Pandoc

**Examples:**
```bash
# Generate Markdown
neural docs model.neural --output documentation.md

# Generate Markdown and PDF
neural docs model.neural --pdf
```

**Requirements:**
- Pandoc (for PDF export)

---

### explain

Explain model predictions using interpretability methods.

**Syntax:**
```bash
neural explain <model_path> [OPTIONS]
```

**Options:**
- `--method, -m` - Method: `shap`, `lime`, `saliency`, `attention`, `feature_importance`, `counterfactual`, `all`
- `--backend, -b` - Backend: `tensorflow`, `pytorch`
- `--data, -d` - Input data file (`.npy` format)
- `--output, -o` - Output directory (default: `explanations`)
- `--num-samples` - Number of samples to explain (default: `10`)
- `--generate-model-card` - Generate model card

**Examples:**
```bash
# SHAP explanations
neural explain model.h5 --method shap --backend tensorflow --data test_data.npy

# All methods
neural explain model.pth --method all --backend pytorch

# With model card
neural explain model.h5 --generate-model-card
```

---

### version

Show version information.

**Syntax:**
```bash
neural version
```

**Output:**
- Neural DSL version
- Python version
- Platform information
- Environment type (local, Kaggle, Colab)
- Installed frameworks (TensorFlow, PyTorch, JAX, Optuna)

---

### help

Show help information.

**Syntax:**
```bash
neural help
neural <command> --help
```

**Examples:**
```bash
# General help
neural help
neural --help

# Command-specific help
neural compile --help
neural debug --help
```

---

## Environment Variables

```bash
NEURAL_SKIP_WELCOME=1     # Skip welcome message
NEURAL_FORCE_CPU=1        # Force CPU mode
TF_CPP_MIN_LOG_LEVEL=3    # Suppress TensorFlow logs
MPLBACKEND=Agg            # Non-interactive matplotlib
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- Other - Command-specific error code

---

## Common Workflows

### Development Workflow

```bash
# 1. Create model
nano model.neural

# 2. Validate
neural compile model.neural --dry-run

# 3. Visualize
neural visualize model.neural

# 4. Debug
neural debug model.neural --dashboard

# 5. Compile and run
neural compile model.neural --backend tensorflow
neural run model.neural
```

### Experiment Workflow

```bash
# 1. Initialize tracking
neural track init my_experiment

# 2. Run with HPO
neural compile model.neural --hpo

# 3. Log results
neural track log --metrics-file results.json

# 4. Compare experiments
neural track list
neural track compare exp_1 exp_2 exp_3
```

### Deployment Workflow

```bash
# 1. Train model
neural run model.neural

# 2. Export optimized model
neural export model.neural --format onnx --optimize --quantize

# 3. Generate documentation
neural docs model.neural --pdf

# 4. Clean artifacts
neural clean --yes
```

---

## Tips and Best Practices

1. **Use feature groups**: Install only what you need with `pip install neural-dsl[hpo]`
2. **Dry-run first**: Always use `--dry-run` to preview before compiling
3. **Visualize early**: Catch shape errors with `neural visualize`
4. **Track experiments**: Use `neural track` for reproducibility
5. **Clean regularly**: Run `neural clean --yes` to remove generated files
6. **Check version**: Ensure compatibility with `neural --version`

---

## Further Reading

- [DSL Language Reference](dsl.md)
- [Getting Started Guide](../GETTING_STARTED.md)
- [Examples](../examples/README.md)
- [Deployment Guide](deployment.md)
- [AGENTS.md](../AGENTS.md) - Development guide
