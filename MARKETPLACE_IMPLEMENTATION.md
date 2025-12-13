# Neural Marketplace Implementation

This document describes the implementation of the Neural Marketplace feature.

## Overview

The Neural Marketplace is a comprehensive model registry and sharing platform for Neural DSL models. It provides:

- **Model Registry**: Storage and management of Neural DSL models with versioning and licensing
- **Semantic Search**: Advanced search using embeddings and similarity matching
- **HuggingFace Hub Integration**: Upload and download models from HuggingFace Hub
- **REST API**: Programmatic access to marketplace functionality
- **Web UI**: Interactive browser-based interface
- **CLI Commands**: Command-line tools for marketplace operations
- **Usage Statistics**: Tracking of downloads, views, and popularity

## Architecture

### Module Structure

```
neural/marketplace/
├── __init__.py                      # Package initialization and exports
├── registry.py                      # Core registry with CRUD operations
├── search.py                        # Semantic search engine
├── api.py                           # REST API with Flask
├── web_ui.py                        # Web interface with Dash
├── huggingface_integration.py       # HuggingFace Hub integration
├── utils.py                         # Utility functions
├── README.md                        # Full documentation
└── QUICK_START.md                   # Quick start guide
```

### Data Storage

```
neural_marketplace_registry/
├── registry_metadata.json           # Main metadata index
├── usage_stats.json                 # Usage statistics
├── embeddings.json                  # Cached semantic embeddings
└── models/                          # Model storage
    ├── author-model-timestamp/
    │   ├── metadata.json            # Model-specific metadata
    │   └── model.neural             # Model file
    └── ...
```

## Components

### 1. Model Registry (`registry.py`)

**Core Features:**
- Model upload with validation
- Model download with integrity checking
- Metadata management (CRUD operations)
- Versioning support
- License tracking
- Usage statistics
- Tag-based organization

**Key Methods:**
- `upload_model()`: Upload a new model
- `download_model()`: Download a model by ID
- `get_model_info()`: Get model metadata
- `list_models()`: List models with filtering
- `update_model()`: Update model metadata
- `delete_model()`: Remove a model
- `get_usage_stats()`: Get usage statistics
- `get_popular_models()`: Get most downloaded models
- `get_recent_models()`: Get recently uploaded models

### 2. Semantic Search (`search.py`)

**Core Features:**
- TF-IDF-style embedding generation
- Cosine similarity-based search
- Architecture-based search
- Task-based search
- Similar model discovery
- Trending tags
- Autocomplete suggestions

**Key Methods:**
- `search()`: Search models by query
- `search_by_architecture()`: Search by architecture type
- `search_by_task()`: Search by task type
- `find_similar_models()`: Find similar models
- `get_trending_tags()`: Get popular tags
- `autocomplete()`: Get search suggestions

### 3. HuggingFace Hub Integration (`huggingface_integration.py`)

**Core Features:**
- Upload models to HuggingFace Hub
- Download models from HuggingFace Hub
- Search Hub for Neural DSL models
- Model card generation
- Repository management

**Key Methods:**
- `upload_to_hub()`: Upload a model to Hub
- `download_from_hub()`: Download from Hub
- `search_hub()`: Search Hub for models
- `get_model_info()`: Get model info from Hub
- `list_user_models()`: List user's models

### 4. REST API (`api.py`)

**Endpoints:**

Model Operations:
- `GET /api/models` - List all models
- `GET /api/models/<id>` - Get model info
- `POST /api/models/upload` - Upload model
- `PUT /api/models/<id>` - Update model
- `DELETE /api/models/<id>` - Delete model
- `POST /api/models/<id>/download` - Download model

Search Operations:
- `GET /api/search?q=query` - Search models
- `GET /api/search/similar/<id>` - Find similar
- `GET /api/search/autocomplete?q=prefix` - Autocomplete

Statistics:
- `GET /api/tags` - Trending tags
- `GET /api/popular` - Popular models
- `GET /api/recent` - Recent models
- `GET /api/stats` - Marketplace statistics

HuggingFace Hub:
- `POST /api/hub/upload` - Upload to Hub
- `POST /api/hub/download` - Download from Hub
- `GET /api/hub/search?q=query` - Search Hub

### 5. Web UI (`web_ui.py`)

**Pages:**
- **Browse**: Search and filter models
- **Upload**: Upload new models
- **My Models**: Manage your models
- **HuggingFace**: Hub integration

**Features:**
- Interactive search with filters
- Model cards with details
- Statistics dashboard
- Trending tags display
- Responsive design

### 6. CLI Commands

**Command Group:** `neural marketplace`

**Commands:**
- `search [query]` - Search for models
- `download <id>` - Download a model
- `publish <path>` - Upload a model
- `info <id>` - Get model information
- `list` - List all models
- `web` - Launch web interface
- `hub-upload <path> <repo>` - Upload to HuggingFace Hub
- `hub-download <repo> <file>` - Download from Hub

### 7. Utilities (`utils.py`)

**Functions:**
- `calculate_file_hash()`: Compute file SHA256
- `validate_license()`: Check license validity
- `format_model_size()`: Human-readable sizes
- `parse_version()`: Parse semantic versions
- `compare_versions()`: Compare versions
- `sanitize_model_name()`: Clean model names
- `extract_model_metadata()`: Parse model files
- `generate_model_card()`: Create model cards
- `validate_model_file()`: Validate model syntax
- `create_backup()`: Backup registry
- `restore_backup()`: Restore from backup

## Data Models

### Model Metadata Schema

```json
{
  "id": "author/model-name-timestamp",
  "name": "Model Name",
  "author": "Author Name",
  "description": "Model description",
  "license": "MIT",
  "tags": ["tag1", "tag2"],
  "framework": "neural-dsl",
  "version": "1.0.0",
  "file": "model.neural",
  "file_hash": "sha256...",
  "uploaded_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00",
  "downloads": 0,
  "metadata": {}
}
```

### Usage Statistics Schema

```json
{
  "model-id": {
    "downloads": 0,
    "views": 0,
    "last_accessed": "2024-01-01T12:00:00"
  }
}
```

### Registry Metadata Schema

```json
{
  "models": {
    "model-id": { /* model metadata */ }
  },
  "tags": {
    "tag-name": ["model-id-1", "model-id-2"]
  },
  "authors": {
    "author-name": ["model-id-1", "model-id-2"]
  }
}
```

## Usage Examples

### Python API

```python
from neural.marketplace import ModelRegistry, SemanticSearch

# Initialize
registry = ModelRegistry()
search = SemanticSearch(registry)

# Upload
model_id = registry.upload_model(
    name="My Model",
    author="John Doe",
    model_path="model.neural",
    tags=["classification"]
)

# Search
results = search.search("classification", limit=10)

# Download
file_path = registry.download_model(model_id)
```

### CLI

```bash
# Publish model
neural marketplace publish model.neural \
  --name "My Model" \
  --author "John Doe" \
  --tags "classification,cnn"

# Search
neural marketplace search "classification"

# Download
neural marketplace download <model-id>

# Launch web UI
neural marketplace web --port 8052
```

### REST API

```bash
# Search models
curl http://localhost:5000/api/search?q=classification

# Get model info
curl http://localhost:5000/api/models/<model-id>

# Get statistics
curl http://localhost:5000/api/stats
```

## Testing

Run tests with:

```bash
pytest tests/test_marketplace.py -v
```

Test coverage includes:
- Model upload/download
- Search functionality
- Metadata management
- Statistics tracking
- Utility functions
- Integration workflows

## Dependencies

**Core:**
- `pyyaml` - Configuration and metadata
- `numpy` - Embeddings and similarity

**Optional:**
- `flask`, `flask-cors` - REST API
- `dash`, `dash-bootstrap-components` - Web UI
- `huggingface_hub` - HuggingFace integration

## Security Considerations

1. **File Validation**: All uploaded files are validated
2. **Hash Verification**: File integrity checked with SHA256
3. **Input Sanitization**: Model names and metadata sanitized
4. **Access Control**: Token-based authentication for Hub operations
5. **Path Safety**: All file operations use safe path handling

## Performance

- **Search**: O(n) where n = number of models (optimized with caching)
- **Upload**: O(1) for metadata, O(file_size) for file copy
- **Download**: O(1) for lookup, O(file_size) for file copy
- **Embeddings**: Cached to avoid recomputation

## Future Enhancements

1. **Advanced Search**: Use transformer-based embeddings
2. **Model Benchmarking**: Track model performance metrics
3. **Social Features**: Comments, ratings, favorites
4. **Collections**: Curated model collections
5. **API Keys**: Authentication for API access
6. **Model Comparison**: Side-by-side model comparison
7. **Export Formats**: Support for ONNX, TFLite, etc.
8. **CI/CD Integration**: Automated testing and deployment
9. **Federated Learning**: Distributed model training
10. **Model Compression**: Automatic model optimization

## Contributing

To contribute to the marketplace:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull requests with clear descriptions

## License

The marketplace implementation is part of Neural DSL and follows the MIT license.

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See README.md in neural/marketplace/
- Examples: See examples/marketplace_demo.py

---

**Implementation Status**: ✅ Complete

**Files Created**:
- neural/marketplace/__init__.py
- neural/marketplace/registry.py
- neural/marketplace/search.py
- neural/marketplace/api.py
- neural/marketplace/web_ui.py
- neural/marketplace/huggingface_integration.py
- neural/marketplace/utils.py
- neural/marketplace/README.md
- neural/marketplace/QUICK_START.md
- examples/marketplace_demo.py
- tests/test_marketplace.py

**CLI Commands Added**: 11 new marketplace commands
**API Endpoints**: 16 REST endpoints
**Web UI Routes**: 4 main pages with interactive features
