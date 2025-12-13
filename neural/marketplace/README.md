# Neural Marketplace

Model marketplace and registry for Neural DSL with semantic search, versioning, licensing, and HuggingFace Hub integration.

## Features

- **Model Registry**: Upload, download, and manage Neural DSL models
- **Semantic Search**: Advanced search using embeddings and similarity matching
- **Versioning**: Track model versions and updates
- **Licensing**: Support for various open-source licenses (MIT, Apache-2.0, GPL-3.0, etc.)
- **Usage Statistics**: Track downloads, views, and popularity
- **HuggingFace Hub Integration**: Upload and download models from HuggingFace Hub
- **REST API**: RESTful API for programmatic access
- **Web UI**: Interactive web interface for browsing and managing models

## Installation

```bash
# Install core marketplace
pip install -e .

# Install with all marketplace dependencies
pip install -e ".[dashboard,ml-extras]"
```

## CLI Usage

### Browse and Search

```bash
# Search for models
neural marketplace search "resnet classification"

# Search with filters
neural marketplace search "cnn" --author "john_doe" --tags "classification,image" --license MIT

# List all models
neural marketplace list

# List with filters
neural marketplace list --author "jane_smith" --sort-by downloads --limit 20

# Get model information
neural marketplace info <model-id>
```

### Upload and Download

```bash
# Publish a model
neural marketplace publish model.neural \
  --name "My ResNet Model" \
  --author "Your Name" \
  --description "A ResNet-based classifier for CIFAR-10" \
  --tags "classification,resnet,cifar10" \
  --license MIT \
  --version 1.0.0

# Download a model
neural marketplace download <model-id> --output ./models/

# Update model metadata
neural marketplace update <model-id> \
  --description "Updated description" \
  --version 1.1.0
```

### HuggingFace Hub Integration

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Upload to HuggingFace Hub
neural marketplace hub-upload model.neural username/repo-name \
  --name "My Model" \
  --description "A great model" \
  --tags "neural-dsl,classification"

# Download from HuggingFace Hub
neural marketplace hub-download username/repo-name model.neural

# Search HuggingFace Hub
neural marketplace hub-search --query "classification"
```

### Web Interface

```bash
# Launch the marketplace web UI
neural marketplace web --port 8052

# Then open http://localhost:8052/marketplace in your browser
```

## Python API Usage

### Model Registry

```python
from neural.marketplace import ModelRegistry

# Initialize registry
registry = ModelRegistry("my_registry")

# Upload a model
model_id = registry.upload_model(
    name="My CNN Model",
    author="John Doe",
    model_path="model.neural",
    description="A CNN for image classification",
    license="MIT",
    tags=["classification", "cnn", "image"],
    version="1.0.0"
)

# Download a model
file_path = registry.download_model(model_id, output_dir="./models")

# Get model info
info = registry.get_model_info(model_id)
print(f"Model: {info['name']} by {info['author']}")
print(f"Downloads: {info['downloads']}")

# List models
models = registry.list_models(
    author="John Doe",
    tags=["classification"],
    sort_by="downloads",
    limit=10
)

# Update model
registry.update_model(
    model_id,
    description="Updated description",
    version="1.1.0",
    tags=["classification", "cnn", "image", "updated"]
)

# Get usage statistics
stats = registry.get_usage_stats(model_id)
print(f"Downloads: {stats['downloads']}, Views: {stats['views']}")
```

### Semantic Search

```python
from neural.marketplace import ModelRegistry, SemanticSearch

# Initialize
registry = ModelRegistry()
search = SemanticSearch(registry)

# Search models
results = search.search(
    query="resnet image classification",
    limit=10,
    filters={"author": "john_doe", "license": "MIT"}
)

for model_id, similarity, model_info in results:
    print(f"{model_info['name']} (similarity: {similarity:.2f})")
    print(f"  Author: {model_info['author']}")
    print(f"  Tags: {', '.join(model_info['tags'])}")

# Search by architecture
results = search.search_by_architecture("ResNet", limit=5)

# Search by task
results = search.search_by_task("classification", limit=5)

# Find similar models
similar = search.find_similar_models(model_id, limit=5)

# Get trending tags
tags = search.get_trending_tags(limit=20)
print("Trending tags:")
for tag, count in tags:
    print(f"  {tag}: {count} models")

# Autocomplete
suggestions = search.autocomplete("class", limit=10)
print(f"Suggestions: {suggestions}")
```

### HuggingFace Hub Integration

```python
from neural.marketplace import HuggingFaceIntegration

# Initialize with token
hf = HuggingFaceIntegration(token="your_hf_token")

# Upload to Hub
result = hf.upload_to_hub(
    model_path="model.neural",
    repo_id="username/my-model",
    model_name="My Model",
    description="A great model for classification",
    license="mit",
    tags=["neural-dsl", "classification"],
    private=False
)
print(f"Uploaded to: {result['url']}")

# Download from Hub
file_path = hf.download_from_hub(
    repo_id="username/my-model",
    filename="model.neural",
    output_dir="./models"
)

# Search Hub
models = hf.search_hub(
    query="classification",
    tags=["neural-dsl"],
    limit=20
)

for model in models:
    print(f"{model['name']} by {model['author']}")
    print(f"  Downloads: {model['downloads']}, Likes: {model['likes']}")

# Get model info
info = hf.get_model_info("username/my-model")
print(f"Model: {info['name']}")
print(f"Tags: {', '.join(info['tags'])}")
```

### REST API

```python
from neural.marketplace import MarketplaceAPI

# Initialize API
api = MarketplaceAPI(registry_dir="my_registry")

# Run API server
api.run(host="0.0.0.0", port=5000, debug=False)
```

API Endpoints:

- `GET /api/models` - List all models
- `GET /api/models/<model_id>` - Get model information
- `POST /api/models/upload` - Upload a model
- `PUT /api/models/<model_id>` - Update model metadata
- `DELETE /api/models/<model_id>` - Delete a model
- `POST /api/models/<model_id>/download` - Download a model
- `GET /api/search?q=query` - Search models
- `GET /api/search/similar/<model_id>` - Find similar models
- `GET /api/search/autocomplete?q=prefix` - Autocomplete suggestions
- `GET /api/tags` - Get trending tags
- `GET /api/popular` - Get popular models
- `GET /api/recent` - Get recent models
- `GET /api/stats` - Get marketplace statistics

HuggingFace Hub endpoints:
- `POST /api/hub/upload` - Upload to HuggingFace Hub
- `POST /api/hub/download` - Download from HuggingFace Hub
- `GET /api/hub/search?q=query` - Search HuggingFace Hub

### Web UI

```python
from neural.marketplace import MarketplaceUI

# Initialize UI
ui = MarketplaceUI(registry_dir="my_registry")

# Run web server
ui.run(host="0.0.0.0", port=8052, debug=False)
```

## Architecture

```
neural/marketplace/
├── __init__.py              # Package exports
├── registry.py              # Model registry with versioning & licensing
├── search.py                # Semantic search engine
├── api.py                   # REST API with Flask
├── web_ui.py                # Web interface with Dash
├── huggingface_integration.py  # HuggingFace Hub integration
└── README.md                # This file

Registry Directory Structure:
neural_marketplace_registry/
├── registry_metadata.json   # Model metadata index
├── usage_stats.json         # Usage statistics
├── embeddings.json          # Cached semantic embeddings
└── models/                  # Model files
    ├── model-id-1/
    │   ├── metadata.json
    │   └── model.neural
    └── model-id-2/
        ├── metadata.json
        └── model.neural
```

## Model Metadata Schema

```json
{
  "id": "author/model-name-20240101120000",
  "name": "My Model",
  "author": "John Doe",
  "description": "A great model for classification",
  "license": "MIT",
  "tags": ["classification", "cnn", "image"],
  "framework": "neural-dsl",
  "version": "1.0.0",
  "file": "model.neural",
  "file_hash": "sha256...",
  "uploaded_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00",
  "downloads": 42,
  "metadata": {
    "architecture": "ResNet",
    "dataset": "CIFAR-10",
    "accuracy": 0.95
  }
}
```

## Supported Licenses

- MIT
- Apache-2.0
- GPL-3.0
- BSD-3-Clause
- BSD-2-Clause
- LGPL-3.0
- MPL-2.0
- AGPL-3.0
- Unlicense
- CC0-1.0
- Custom (specify in metadata)

## Best Practices

1. **Use descriptive names**: Choose clear, descriptive names for your models
2. **Add detailed descriptions**: Help others understand what your model does
3. **Tag appropriately**: Use relevant tags for better discoverability
4. **Version semantically**: Follow semantic versioning (MAJOR.MINOR.PATCH)
5. **Choose appropriate licenses**: Select licenses that match your sharing goals
6. **Update regularly**: Keep model information and versions up to date
7. **Document thoroughly**: Include usage examples and requirements

## Examples

See the `examples/marketplace/` directory for complete examples:

- `basic_usage.py` - Basic registry operations
- `search_demo.py` - Semantic search examples
- `api_client.py` - REST API client examples
- `hub_integration.py` - HuggingFace Hub integration examples

## Contributing

Contributions are welcome! Please see the main repository's CONTRIBUTING.md for guidelines.

## License

This marketplace implementation is part of Neural DSL and follows the same MIT license.
