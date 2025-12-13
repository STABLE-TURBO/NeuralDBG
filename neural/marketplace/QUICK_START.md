# Neural Marketplace Quick Start

Get started with the Neural Marketplace in 5 minutes!

## Installation

```bash
# Install Neural DSL with marketplace dependencies
pip install -e ".[dashboard,ml-extras]"

# For HuggingFace Hub integration
pip install huggingface_hub
```

## Quick Examples

### 1. Publish Your First Model

```bash
# Create a simple model
cat > my_model.neural << EOF
Network MyClassifier {
    Input: shape=(28, 28, 1)
    Conv2D: filters=32, kernel=(3,3), activation=relu
    MaxPooling2D: pool_size=(2,2)
    Flatten
    Dense: units=128, activation=relu
    Dense: units=10, activation=softmax
    Output: loss=categorical_crossentropy
}
EOF

# Publish to marketplace
neural marketplace publish my_model.neural \
  --name "My First Classifier" \
  --author "Your Name" \
  --description "A simple CNN for MNIST classification" \
  --tags "classification,mnist,cnn" \
  --license MIT
```

### 2. Search and Download Models

```bash
# Search for models
neural marketplace search "classification"

# Get model details
neural marketplace info <model-id>

# Download a model
neural marketplace download <model-id>
```

### 3. Browse in Web UI

```bash
# Launch the web interface
neural marketplace web --port 8052

# Open http://localhost:8052/marketplace in your browser
```

### 4. Use Python API

```python
from neural.marketplace import ModelRegistry, SemanticSearch

# Initialize
registry = ModelRegistry()
search = SemanticSearch(registry)

# Upload a model
model_id = registry.upload_model(
    name="My Model",
    author="Your Name",
    model_path="model.neural",
    description="A great model",
    tags=["classification"]
)

# Search for models
results = search.search("classification cnn", limit=5)
for model_id, similarity, model in results:
    print(f"{model['name']}: {similarity:.2f}")

# Download a model
file_path = registry.download_model(model_id)
```

### 5. HuggingFace Hub Integration

```bash
# Set your token
export HF_TOKEN=your_token_here

# Upload to HuggingFace Hub
neural marketplace hub-upload my_model.neural username/repo-name \
  --name "My Model" \
  --description "A great model" \
  --tags "neural-dsl,classification"

# Download from HuggingFace Hub
neural marketplace hub-download username/repo-name model.neural
```

## Common Tasks

### List All Models

```bash
neural marketplace list --sort-by downloads --limit 20
```

### Filter by Author

```bash
neural marketplace list --author "John Doe"
```

### Filter by Tags

```bash
neural marketplace search --tags "classification,cnn"
```

### Update Model Metadata

```python
from neural.marketplace import ModelRegistry

registry = ModelRegistry()
registry.update_model(
    model_id="author/model-name-...",
    description="Updated description",
    version="1.1.0",
    tags=["classification", "updated"]
)
```

### Get Statistics

```python
from neural.marketplace import ModelRegistry

registry = ModelRegistry()

# Get overall stats
total_models = len(registry.metadata["models"])
total_downloads = sum(s.get("downloads", 0) for s in registry.stats.values())

# Get model-specific stats
stats = registry.get_usage_stats(model_id)
print(f"Downloads: {stats['downloads']}, Views: {stats['views']}")

# Get popular models
popular = registry.get_popular_models(limit=10)
```

## Environment Variables

```bash
# HuggingFace token for Hub operations
export HF_TOKEN=your_token_here

# Custom registry directory
export NEURAL_MARKETPLACE_DIR=path/to/registry
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Run the demo: `python examples/marketplace_demo.py`
- Explore the REST API at http://localhost:5000/api/
- Check out example models in the marketplace
- Share your models with the community!

## Troubleshooting

### Import Error

```bash
# Install missing dependencies
pip install dash dash-bootstrap-components flask flask-cors
```

### HuggingFace Hub Issues

```bash
# Install HuggingFace Hub
pip install huggingface_hub

# Set your token
export HF_TOKEN=your_token_here

# Test authentication
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### Registry Permission Error

```bash
# Check directory permissions
ls -la neural_marketplace_registry/

# Use a custom directory
neural marketplace publish model.neural --name "..." --author "..." \
  --registry-dir ~/my_registry/
```

## Support

For help and issues:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See README.md and examples/
- CLI help: `neural marketplace --help`
