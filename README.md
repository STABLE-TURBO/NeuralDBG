<div align="center">
  <img src="https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b" alt="Neural Logo" width="200"/>
  <h1>Neural</h1>
  <p>A domain-specific language for defining neural networks</p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
  [![CI](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml)
  [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lemniscate-world.github.io/Neural/)
  [![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/KFku4KvS)
</div>

---

## Why This Exists

I got tired of writing the same boilerplate over and over. You know the drill: import torch, define a class, write a forward method, manually calculate flattened dimensions after conv layers, copy-paste training loops, and then spend an hour debugging shape mismatches that show up at runtime.

One weekend I thought: what if I could just declare what I want the model to be, and let something else handle the tedious parts? That's how Neural started. It's a DSL (domain-specific language) for neural networks that reads like a config file but gives you real Python code for TensorFlow, PyTorch, or ONNX.

The core idea: write your model once in a simple, declarative syntax, and compile it to whatever framework you need. Plus, catch shape errors before you hit run, and get a debugging dashboard without writing custom visualization code.

![Neural Demo](https://github.com/user-attachments/assets/ecbcce19-73df-4696-ace2-69e32d02709f)

## Quick Example

Here's a CNN for MNIST in Neural DSL:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 10
    batch_size: 64
  }
}
```

That's it. No imports, no manual dimension calculations, no training loop boilerplate. Run `neural compile mnist.neural --backend pytorch` and you get working PyTorch code. Switch to `--backend tensorflow` and you get TensorFlow instead.

Compare that to the ~100 lines of PyTorch you'd normally write (model class, training loop, data loading, etc.).

## The Pain Points It Actually Solves

**Shape mismatches at runtime**: We've all been there. You run your model, wait for data to load, start training, and then boom—dimension mismatch on line 47. Neural validates shapes before you run anything and shows you exactly where dimensions don't line up.

**Switching frameworks is a nightmare**: Need to move from PyTorch to TensorFlow for deployment? That's usually a multi-day rewrite. With Neural, it's a flag change.

**Debugging is tedious**: Setting up TensorBoard or writing custom hooks to inspect gradients, activations, and memory usage takes forever. Neural includes NeuralDbg—a dashboard that tracks all this automatically. Just run `neural debug model.neural` and open `localhost:8050`.

**Boilerplate everywhere**: Training loops, data preprocessing, model checkpointing... it's the same code every time with slight variations. Neural handles this so you can focus on architecture and hyperparameters.

**Framework lock-in**: Once you commit to a framework, you're kinda stuck. Neural lets you stay framework-agnostic until you need to be specific.

## What's Included

### Shape Validation
The DSL parser propagates tensor shapes through your entire model and tells you if dimensions don't match. No more runtime surprises.

```bash
neural visualize mnist.neural --format png
```

You get diagrams showing tensor transformations at each layer. It's especially helpful for debugging conv/pooling stacks where calculating output dimensions manually is annoying.

![Shape Propagation](https://github.com/user-attachments/assets/5c4f51b5-e40f-47b3-872d-445f71c6582f)

### Cross-Framework Compilation
Same DSL code, multiple backends:

```bash
neural compile model.neural --backend tensorflow --output tf_model.py
neural compile model.neural --backend pytorch --output torch_model.py
neural compile model.neural --backend onnx --output model.onnx
```

The generated code is readable Python that you can modify if needed. Neural doesn't lock you into some opaque abstraction layer.

### Real-Time Debugging (NeuralDbg)
Start the debugger with:

```bash
neural debug model.neural
```

Then open the dashboard at `http://localhost:8050`. You'll see:

- Execution traces showing which layers ran and how long they took
- Gradient flow visualization (helps catch vanishing/exploding gradients)
- Dead neuron detection (shows you which units never activate)
- Memory and FLOPs profiling
- Anomaly detection for NaN/Inf values

<div align="center">

**Execution Trace**
![test_trace_graph](https://github.com/user-attachments/assets/15b1edd2-2643-4587-9843-aa4697ed2e4b)

**Gradient Flow**
![test_gradient_chart](https://github.com/user-attachments/assets/ca6b9f20-7dd8-4c72-9ee8-a3f35af6208b)

</div>

I originally built this because I was tired of writing custom TensorBoard logging for every experiment. NeuralDbg isn't perfect, but it covers 90% of what I need when debugging a new architecture.

### Experiment Tracking
Every time you run a model, Neural logs hyperparameters, metrics, and training time to a local SQLite database. Then you can compare runs:

```bash
neural track list                    # show all experiments
neural track show <experiment_id>    # details for one run
neural track compare exp_1 exp_2     # side-by-side comparison
neural track plot exp_1              # plot metrics over time
```

It's simpler than MLflow for quick local experiments, though you can still integrate with MLflow if you want.

### Deployment Export
When you're ready to deploy, Neural can export optimized models:

```bash
# Export to ONNX (cross-platform inference)
neural export model.neural --format onnx --optimize

# TensorFlow Lite with quantization (mobile/edge)
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type int8

# TorchScript (PyTorch production)
neural export model.neural --backend pytorch --format torchscript

# TensorFlow Serving
neural export model.neural --backend tensorflow --format savedmodel --deployment tfserving
```

See [docs/deployment.md](docs/deployment.md) for more details on deployment options.

## Installation

### For End Users

```bash
pip install neural-dsl[full]
```

The `[full]` extra installs TensorFlow, PyTorch, ONNX, and all optional features (~2.5 GB). If you only need specific parts:

```bash
pip install neural-dsl              # core DSL parsing only (~20 MB)
pip install neural-dsl[backends]    # add TensorFlow/PyTorch/ONNX
pip install neural-dsl[hpo]         # hyperparameter optimization with Optuna
pip install neural-dsl[automl]      # automated ML and NAS
pip install neural-dsl[dashboard]   # NeuralDbg interface
pip install neural-dsl[visualization] # charts and diagrams
pip install neural-dsl[cloud]       # cloud integrations
pip install neural-dsl[integrations] # ML platform connectors
```

### For Development

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate              # Windows
# source .venv/bin/activate           # Linux/macOS

# Install development dependencies (includes core package in editable mode)
pip install -r requirements-dev.txt

# Optional: Set up pre-commit hooks
pre-commit install
```

For more detailed installation instructions including profiles, troubleshooting, and platform-specific notes, see [INSTALL.md](INSTALL.md).

## Getting Started

1. **Write a model** in `.neural` syntax (see [examples/](examples/) for templates)
2. **Validate it**: `neural visualize model.neural`
3. **Compile and run**: `neural run model.neural --backend tensorflow`
4. **Debug if needed**: `neural debug model.neural`
5. **Export for production**: `neural export model.neural --format onnx`

Full command reference:

```bash
neural compile <file>       # Generate Python code
neural run <file>           # Compile + execute training
neural visualize <file>     # Generate architecture diagrams
neural debug <file>         # Start debugging dashboard
neural export <file>        # Export for deployment
neural track <command>      # Manage experiments
neural --no_code            # Launch no-code GUI
```

## What It's Good At (and What It's Not)

**Good for:**
- Prototyping standard architectures quickly
- Teaching/learning neural network concepts
- Comparing frameworks side-by-side
- Catching shape errors early
- Simple deployment workflows

**Not great for:**
- Cutting-edge research with custom ops (the DSL can't express everything yet)
- Highly optimized production code (generated code is readable but not maximally performant)
- Very large models (shape propagation can be slow on 1000+ layer architectures)
- Dynamic architectures like conditional computation (DSL is declarative, so dynamic control flow is limited)

The DSL syntax covers common layers (Conv, Dense, LSTM, Transformer blocks, etc.) but you'll hit limits with exotic custom layers. When that happens, you can generate code as a starting point and then modify the Python directly.

## Known Limitations

- **Type checking**: The DSL parser does shape validation but doesn't catch all type mismatches (e.g., mixing float32/float64 inconsistently)
- **Error messages**: Sometimes cryptic. I'm working on improving this, but for now you might need to look at generated code to debug
- **Performance overhead**: Generated code is ~0-20% slower than hand-written equivalents due to extra abstraction. Usually negligible, but matters for production serving at scale
- **Backend coverage**: Not all DSL features work on all backends. For example, some custom layers only compile to TensorFlow right now
- **No distributed training**: Multi-GPU and distributed setups aren't supported yet (planned for future versions)
- **Windows quirks**: NeuralDbg dashboard sometimes has issues on Windows. Works best on Linux/macOS

If you hit a limitation, please open an issue. I'm actively developing this and prioritize based on user feedback.

## AI Model Generation (Experimental)

v0.3.0 added a feature to generate DSL code from natural language. It's still experimental but can be useful:

```python
from neural.ai import generate_model

model_code = generate_model("""
    Build a CNN for MNIST digit classification.
    Use 2 conv layers with 32 and 64 filters.
    Add dropout and dense layers for classification.
""")

with open("generated.neural", "w") as f:
    f.write(model_code)
```

This uses a language model under the hood (requires API key setup—see [docs/ai_integration_guide.md](docs/ai_integration_guide.md)). It works best for standard architectures and sometimes hallucinates invalid syntax for complex models. Always review generated code before running.

## Documentation

**📚 [Full Documentation on GitHub Pages](https://lemniscate-world.github.io/Neural/)**

Quick references:
- [DSL Language Reference](docs/dsl.md) – Complete syntax guide
- [Deployment Guide](docs/deployment.md) – Production export options
- [AI Integration Guide](docs/ai_integration_guide.md) – Natural language model generation

### Quick References
- [Transformer Reference](docs/transformer_reference.md) – TransformerEncoder quick reference
- [Automation Reference](docs/AUTOMATION_REFERENCE.md) – Release and distribution automation
- [Documentation Index](docs/DOCUMENTATION_INDEX.md) – Complete documentation guide

### Contributing
- [Contributing Guide](CONTRIBUTING.md) – How to contribute
- [Agents Guide](AGENTS.md) – Repository setup for development

More examples in the [examples/](examples/) directory.

## Community

- **Discord**: [Join the server](https://discord.gg/KFku4KvS) for questions and discussion
- **GitHub Discussions**: [Share ideas or ask for help](https://github.com/Lemniscate-world/Neural/discussions)
- **Twitter**: [@NLang4438](https://x.com/NLang4438) for updates
- **Issues**: [Report bugs here](https://github.com/Lemniscate-world/Neural/issues)

I'm pretty responsive on Discord if you want quick feedback.

## Contributing

Contributions are welcome. Whether it's fixing bugs, adding examples, improving docs, or implementing features—everything helps.

To get started:

```bash
git clone https://github.com/YOUR_USERNAME/Neural.git
cd Neural
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate              # Windows
# source .venv/bin/activate           # Linux/macOS

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

Check out [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. Look for [good first issue](https://github.com/Lemniscate-world/Neural/labels/good%20first%20issue) tags if you're new.

## Repository Hygiene

The repository recently underwent a major cleanup to remove 200+ redundant files. To maintain cleanliness:

```bash
# Quick cleanup (Windows)
.\cleanup.bat

# Quick cleanup (Linux/macOS)
make clean-dry    # Preview
make clean        # Execute

# Or use scripts directly
python scripts/cleanup_repository.py
.\scripts\cleanup_repository.ps1 -DryRun
bash scripts/cleanup_repository.sh --dry-run
```

See [REPOSITORY_HYGIENE.md](REPOSITORY_HYGIENE.md) for guidelines on what to avoid (implementation summaries, quick reference docs, temp scripts, etc.).

## Development Workflow

Quick reference for common dev tasks:

```bash
# Lint
python -m ruff check .

# Type check (fast, scoped)
python -m mypy neural/code_generation neural/utils

# Run tests
python -m pytest tests/ -v

# Security audit
python -m pip_audit -l --progress-spinner off
```

See the Development Workflow section at the end of this README for full details.

## What's Next

Current focus areas:
- Improve error messages (especially for shape mismatches)
- Expand DSL syntax for more layer types (custom attention, graph convolutions, etc.)
- Add distributed training support (multi-GPU)
- Better Windows support for NeuralDbg
- More deployment targets (CoreML, TensorRT)

See [ROADMAP.md](ROADMAP.md) for the full plan.

## License

MIT License. See [LICENSE](LICENSE.md) for details.

## Acknowledgments

Thanks to everyone who's contributed, filed issues, or just tried Neural and gave feedback. This project exists because people actually use it.

Special thanks to the Lark parsing library (which makes the DSL parsing possible) and to the communities around TensorFlow, PyTorch, and ONNX for building great frameworks to target.

---

## Architecture

<div align="center">

**Class Diagram**
![classes](https://github.com/Lemniscate-world/Neural/blob/main/classes.png)

**Package Structure**
![packages](https://github.com/Lemniscate-world/Neural/blob/main/packages.png)

</div>

Repository structure:

```
neural/
├── cli/                # Command-line interface
├── parser/             # DSL parser (Lark-based)
├── code_generation/    # Code generators for TF/PyTorch/ONNX
├── shape_propagation/  # Shape validation logic
├── dashboard/          # NeuralDbg debugger
├── hpo/                # Hyperparameter optimization
├── cloud/              # Cloud platform integrations
├── tracking/           # Experiment tracking
└── no_code/            # No-code web interface

examples/               # Example .neural files
docs/                   # Documentation
tests/                  # Test suite
```

---

## Development Workflow

This section outlines a minimal, fast local workflow to lint, type-check, test, and audit changes before opening a PR.

### 1) Environment setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

Install the project with development dependencies:

```powershell
pip install -r requirements-dev.txt
```

This installs the core package in editable mode plus all development tools (ruff, mypy, pylint, pytest, pre-commit, pip-audit).

### 2) Common checks (fast)

**Lint (Ruff)**

```powershell
python -m ruff check .
```

**Type check (mypy)**

Fast, scoped type check for currently-hardened modules:
```powershell
python -m mypy neural/code_generation neural/utils
```

Full project type check (may show many findings; tighten gradually):
```powershell
python -m mypy .
```

**Tests (targeted and full)**

Run fast, targeted tests:
```powershell
python -m pytest -q tests/test_seed.py tests/code_generator/test_policy_and_parity.py tests/code_generator/test_policy_helpers.py -rA
```

Run full test suite (may require optional deps such as torch/tensorflow/onnx):
```powershell
python -m pytest -q -rA
```

**Supply-chain audit**

```powershell
python -m pip_audit -l --progress-spinner off
```

### 3) Commit & PR hygiene

- Keep PRs small and focused; include context in the description.
- Run lint, type check (scoped or full), tests, and pip-audit locally before pushing.
- Do not commit secrets/keys. Use environment variables; keep .env or credentials out of Git.
- Follow the shape/policy rules in codegen; add or update tests for any policy changes.

### 4) Optional dependencies for testing

Install only what you need for the tests you are running (examples):

```powershell
# PyTorch backend tests
pip install neural-dsl[backends]

# Or install specific backends individually
pip install torch           # PyTorch only
pip install tensorflow      # TensorFlow only
pip install onnx            # ONNX only

# HPO tests
pip install neural-dsl[hpo]

# Dashboard tests
pip install neural-dsl[dashboard]

# Full feature set (for comprehensive testing)
pip install neural-dsl[full]
```

If you have questions or want guidance on tightening typing or adding new policy checks, open a discussion or draft PR.
