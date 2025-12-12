# Neural DSL Documentation

This directory contains the complete documentation for Neural DSL, including user guides, API reference, and development documentation.

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── requirements.txt       # Documentation build dependencies
├── Makefile              # Unix/Linux/Mac build scripts
├── make.bat              # Windows build scripts
├── BUILD_DOCS.md         # Build instructions
├── API_DOCUMENTATION.md  # API documentation guide
├── DOCSTRING_GUIDE.md    # Docstring style guide
├── api/                  # API reference documentation
│   ├── index.rst
│   ├── neural.rst
│   ├── parser.rst
│   ├── code_generation.rst
│   ├── shape_propagation.rst
│   ├── cli.rst
│   ├── dashboard.rst
│   ├── hpo.rst
│   ├── cloud.rst
│   ├── utils.rst
│   ├── visualization.rst
│   └── README.md
├── _static/              # Static files (CSS, images)
├── _templates/           # Custom Sphinx templates
└── _build/               # Generated documentation (gitignored)
```

## User Documentation

### 1. Getting Started

- [Installation Guide](installation.md) - Setup and dependencies
- [Quick Start Tutorial](tutorials/quickstart_tutorial.ipynb) - Interactive first model (NEW!)
- [Troubleshooting Guide](troubleshooting.md) - Fix common issues (NEW!)
- [Migration Guide](migration.md) - Version upgrades and framework migration (NEW!)

### 2. Neural DSL Reference

- [DSL Syntax](DSL.md)
- [Layer Reference](layers.md)
- [Optimizer Reference](optimizers.md)
- [Training Configuration](training.md)
- [Hyperparameter Specification](hyperparameters.md)

### 3. CLI Reference

- [Command-Line Interface](cli-reference.md)
- [Command Reference](commands.md)
- [Configuration Options](configuration.md)
- [Environment Variables](environment.md)

### 4. API Reference

- [Parser API](api/parser.md)
- [Code Generation API](api/code-generation.md)
- [Shape Propagation API](api/shape-propagation.md)
- [Visualization API](api/visualization.md)
- [Dashboard API](api/dashboard.md)
- [HPO API](api/hpo.md)

### 5. Tutorials

Comprehensive learning resources for all skill levels:

- **[Tutorial Hub](tutorials/README.md)** - Complete tutorial directory with learning paths
- **Interactive Notebooks:**
  - [Quickstart Tutorial](tutorials/quickstart_tutorial.ipynb) - Your first model in 15 minutes (NEW!)
  - [HPO Tutorial](tutorials/hpo_tutorial.ipynb) - Hyperparameter optimization (NEW!)
- **Video Tutorials:**
  - [Video Scripts & Storyboards](tutorials/video_scripts.md) - Complete production guide (NEW!)
  - Getting Started (5 min) - Coming soon
  - Building Your First Model (10 min) - Coming soon
  - Hyperparameter Optimization (8 min) - Coming soon
  - Debugging with NeuralDbg (7 min) - Coming soon
- **Annotated Examples:**
  - [MNIST with Comments](../examples/mnist_commented.neural) - Line-by-line CNN guide (NEW!)
  - [Sentiment Analysis with Comments](../examples/sentiment_analysis_commented.neural) - LSTM tutorial (NEW!)

### 6. Examples

- [Basic Examples](examples/basic/)
- [Computer Vision Examples](examples/computer-vision/)
- [Natural Language Processing Examples](examples/nlp/)
- [Reinforcement Learning Examples](examples/reinforcement-learning/)
- [Generative Models Examples](examples/generative/)

### 7. Guides

- [Best Practices](guides/best-practices.md)
- [Performance Optimization](guides/performance.md)
- [Debugging Guide](guides/debugging.md)
- [Deployment Guide](guides/deployment.md)
- [Contributing Guide](guides/contributing.md)

### 8. Blog

- [Release Notes](blog/releases/)
- [Feature Spotlights](blog/features/)
- [Case Studies](blog/case-studies/)
- [Tutorials](blog/tutorials/)

## Documentation Formats

The documentation is available in multiple formats:

- **Markdown**: The primary format for all documentation
- **HTML**: Generated from Markdown for web viewing
- **PDF**: Generated from Markdown for offline reading
- **Interactive Notebooks**: Jupyter notebooks for tutorials and examples

## Quick Start

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html  # Unix/Linux/Mac
.\make.bat html  # Windows
```

### View Documentation

```bash
open _build/html/index.html  # Mac
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## Writing Documentation

### Adding API Documentation

1. Write NumPy-style docstrings in your code:
   ```python
   def my_function(param: int) -> str:
       """
       Brief description.
       
       Parameters
       ----------
       param : int
           Parameter description
           
       Returns
       -------
       str
           Return value description
       """
   ```

2. Add module to appropriate `.rst` file in `api/`

3. Rebuild documentation

### Docstring Style

Follow the NumPy docstring convention:
- Brief one-line description
- Parameters section with types
- Returns section with type
- Examples section with code
- See `DOCSTRING_GUIDE.md` for details

### Building Locally

```bash
# Clean build
make clean
make html

# Live preview with auto-reload
pip install sphinx-autobuild
sphinx-autobuild . _build/html
# Opens at http://localhost:8000
```

## Documentation Standards

### Quality Checklist

- [ ] All public APIs documented
- [ ] Docstrings follow NumPy style
- [ ] Type hints on all parameters
- [ ] Examples provided where helpful
- [ ] No Sphinx build warnings
- [ ] Links between modules work
- [ ] Code examples are tested

### Testing Documentation

```bash
# Build and check for errors
cd docs
make html

# Check docstring coverage
pip install interrogate
interrogate neural/

# Validate docstrings
pip install pydocstyle
pydocstyle neural/
```

## Contributing to Documentation

We welcome contributions to the documentation! Here's how you can help:

1. **Fix Typos and Errors**: If you find a typo or error, please submit a pull request with the fix.
2. **Improve Existing Documentation**: If you think a section could be clearer or more detailed, feel free to improve it.
3. **Add New Documentation**: If you'd like to add new tutorials, examples, or guides, please submit a pull request.
4. **Translate Documentation**: Help make Neural accessible to more people by translating documentation.

Please follow these guidelines when contributing:

- Use clear, concise language
- Include code examples where appropriate
- Add diagrams and images to illustrate complex concepts
- Follow the existing documentation structure
- Test code examples to ensure they work

When contributing code:

1. **Always add docstrings** to public functions/classes
2. **Follow NumPy style** for consistency
3. **Include type hints** in signatures
4. **Test your examples** to ensure they work
5. **Update .rst files** if adding new modules
6. **Build docs locally** before submitting PR

See `DOCSTRING_GUIDE.md` for detailed guidelines.

## Documentation Tools

The documentation is built using the following tools:

- **MkDocs**: Static site generator for documentation
- **Material for MkDocs**: Theme for MkDocs
- **Mermaid**: Diagramming and charting tool
- **Jupyter Book**: For interactive notebooks
- **Sphinx**: For API documentation

## Continuous Integration

Documentation is automatically built and published:
- On every push to main branch
- Published to Read the Docs (if configured)
- Checked for warnings in CI

## Troubleshooting

### Import Errors
```bash
# Ensure Neural DSL is installed
pip install -e .
```

### Missing Dependencies
```bash
# Install documentation dependencies
pip install -e ".[docs]"
```

### Build Warnings
Check the warning messages and fix:
- Missing docstrings
- Broken cross-references
- Invalid reStructuredText syntax

### Theme Not Found
```bash
pip install sphinx-rtd-theme
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## Support

For documentation issues:
- Open an issue on GitHub
- Check existing documentation
- Refer to Sphinx documentation

## License

Documentation is released under the same license as Neural DSL (MIT License).
