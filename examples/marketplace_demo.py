"""
Neural Marketplace Demo - Example usage of the model marketplace features.
"""

import os
import sys


# Add neural to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural.marketplace import HuggingFaceIntegration, ModelRegistry, SemanticSearch


def demo_registry():
    """Demonstrate model registry operations."""
    print("=" * 60)
    print("MODEL REGISTRY DEMO")
    print("=" * 60)

    # Initialize registry
    registry = ModelRegistry("demo_registry")

    # Create a sample model file
    sample_model = """
Network ResNetDemo {
    Input: shape=(32, 32, 3)
    
    Conv2D: filters=64, kernel=(3,3), activation=relu
    BatchNormalization
    MaxPooling2D: pool_size=(2,2)
    
    Conv2D: filters=128, kernel=(3,3), activation=relu
    BatchNormalization
    MaxPooling2D: pool_size=(2,2)
    
    Flatten
    Dense: units=256, activation=relu
    Dropout: rate=0.5
    Dense: units=10, activation=softmax
    
    Output: loss=categorical_crossentropy, optimizer=adam, metrics=[accuracy]
}
"""

    # Write sample model
    with open("demo_model.neural", "w") as f:
        f.write(sample_model)

    # Upload model
    print("\n1. Uploading model...")
    model_id = registry.upload_model(
        name="ResNet Demo Model",
        author="Demo User",
        model_path="demo_model.neural",
        description="A demonstration ResNet model for CIFAR-10 classification",
        license="MIT",
        tags=["classification", "resnet", "cifar10", "demo"],
        version="1.0.0",
        metadata={
            "architecture": "ResNet",
            "dataset": "CIFAR-10",
            "input_shape": [32, 32, 3],
            "num_classes": 10
        }
    )
    print(f"✓ Model uploaded with ID: {model_id}")

    # Get model info
    print("\n2. Getting model information...")
    info = registry.get_model_info(model_id)
    print(f"✓ Name: {info['name']}")
    print(f"  Author: {info['author']}")
    print(f"  Version: {info['version']}")
    print(f"  License: {info['license']}")
    print(f"  Tags: {', '.join(info['tags'])}")

    # List models
    print("\n3. Listing all models...")
    models = registry.list_models(limit=10)
    print(f"✓ Found {len(models)} models")
    for model in models:
        print(f"  - {model['name']} by {model['author']} (v{model['version']})")

    # Download model
    print("\n4. Downloading model...")
    downloaded_path = registry.download_model(model_id, "downloaded_models")
    print(f"✓ Model downloaded to: {downloaded_path}")

    # Update model
    print("\n5. Updating model metadata...")
    registry.update_model(
        model_id,
        description="Updated: A demonstration ResNet model for CIFAR-10 classification with batch normalization",
        version="1.1.0",
        tags=["classification", "resnet", "cifar10", "demo", "batch-norm"]
    )
    print("✓ Model updated")

    # Get statistics
    print("\n6. Getting usage statistics...")
    stats = registry.get_usage_stats(model_id)
    print(f"✓ Downloads: {stats['downloads']}")
    print(f"  Views: {stats['views']}")

    # Get popular models
    print("\n7. Getting popular models...")
    popular = registry.get_popular_models(limit=5)
    print(f"✓ Top {len(popular)} popular models:")
    for i, model in enumerate(popular, 1):
        print(f"  {i}. {model['name']} ({model['downloads']} downloads)")

    # Cleanup
    os.remove("demo_model.neural")
    if os.path.exists("downloaded_models"):
        import shutil
        shutil.rmtree("downloaded_models")


def demo_search():
    """Demonstrate semantic search operations."""
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH DEMO")
    print("=" * 60)

    # Initialize registry and search
    registry = ModelRegistry("demo_registry")
    search = SemanticSearch(registry)

    # Create some sample models for search
    print("\n1. Creating sample models...")

    sample_models = [
        {
            "name": "MobileNet Classifier",
            "author": "Demo User",
            "description": "Lightweight mobile-optimized CNN for image classification",
            "tags": ["classification", "mobilenet", "mobile", "efficient"],
        },
        {
            "name": "BERT Text Classifier",
            "author": "Demo User",
            "description": "Transformer-based model for text classification using BERT",
            "tags": ["classification", "nlp", "transformer", "bert", "text"],
        },
        {
            "name": "YOLO Object Detector",
            "author": "Demo User",
            "description": "Real-time object detection using YOLO architecture",
            "tags": ["detection", "yolo", "real-time", "object-detection"],
        },
    ]

    for i, model_data in enumerate(sample_models):
        # Create model file
        with open(f"search_demo_{i}.neural", "w") as f:
            f.write(f"Network {model_data['name']} {{\n    Input: shape=(224,224,3)\n}}")

        # Upload to registry
        registry.upload_model(
            name=model_data["name"],
            author=model_data["author"],
            model_path=f"search_demo_{i}.neural",
            description=model_data["description"],
            tags=model_data["tags"]
        )
        os.remove(f"search_demo_{i}.neural")

    print(f"✓ Created {len(sample_models)} sample models")

    # Search for models
    print("\n2. Searching for 'classification'...")
    results = search.search("classification", limit=10)
    print(f"✓ Found {len(results)} results")
    for model_id, similarity, model in results[:3]:
        print(f"  - {model['name']} (similarity: {similarity:.2f})")
        print(f"    {model['description'][:60]}...")

    # Search by architecture
    print("\n3. Searching by architecture 'mobilenet'...")
    results = search.search_by_architecture("mobilenet", limit=5)
    print(f"✓ Found {len(results)} results")
    for model_id, similarity, model in results:
        print(f"  - {model['name']} (similarity: {similarity:.2f})")

    # Search by task
    print("\n4. Searching by task 'detection'...")
    results = search.search_by_task("detection", limit=5)
    print(f"✓ Found {len(results)} results")
    for model_id, similarity, model in results:
        print(f"  - {model['name']} (similarity: {similarity:.2f})")

    # Get trending tags
    print("\n5. Getting trending tags...")
    tags = search.get_trending_tags(limit=10)
    print("✓ Trending tags:")
    for tag, count in tags:
        print(f"  - {tag}: {count} models")

    # Autocomplete
    print("\n6. Autocomplete for 'class'...")
    suggestions = search.autocomplete("class", limit=5)
    print(f"✓ Suggestions: {', '.join(suggestions)}")


def demo_huggingface():
    """Demonstrate HuggingFace Hub integration (requires token)."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE HUB DEMO")
    print("=" * 60)

    # Check if token is available
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠ HuggingFace token not found. Set HF_TOKEN environment variable.")
        print("  Example: export HF_TOKEN=your_token_here")
        print("\nShowing mock operations...")

        print("\n1. Upload to HuggingFace Hub:")
        print("   hf.upload_to_hub(")
        print("       model_path='model.neural',")
        print("       repo_id='username/my-model',")
        print("       model_name='My Model',")
        print("       tags=['neural-dsl', 'classification']")
        print("   )")

        print("\n2. Download from HuggingFace Hub:")
        print("   hf.download_from_hub(")
        print("       repo_id='username/my-model',")
        print("       filename='model.neural'")
        print("   )")

        print("\n3. Search HuggingFace Hub:")
        print("   hf.search_hub(query='classification', tags=['neural-dsl'])")

        return

    try:
        # Initialize HuggingFace integration
        hf = HuggingFaceIntegration(token=hf_token)

        # Search Hub
        print("\n1. Searching HuggingFace Hub for Neural DSL models...")
        models = hf.search_hub(tags=["neural-dsl"], limit=5)
        print(f"✓ Found {len(models)} models")
        for model in models:
            print(f"  - {model['name']} by {model['author']}")
            print(f"    Downloads: {model['downloads']}, Likes: {model['likes']}")

        print("\n✓ HuggingFace Hub integration working!")
        print("  To upload models, use: neural marketplace hub-upload")

    except ImportError:
        print("\n⚠ huggingface_hub package not installed")
        print("  Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"\n⚠ Error: {e}")


def demo_statistics():
    """Demonstrate marketplace statistics."""
    print("\n" + "=" * 60)
    print("MARKETPLACE STATISTICS DEMO")
    print("=" * 60)

    registry = ModelRegistry("demo_registry")

    # Get overall statistics
    print("\n1. Marketplace statistics:")
    total_models = len(registry.metadata["models"])
    total_authors = len(registry.metadata["authors"])
    total_tags = len(registry.metadata["tags"])
    total_downloads = sum(s.get("downloads", 0) for s in registry.stats.values())
    total_views = sum(s.get("views", 0) for s in registry.stats.values())

    print(f"✓ Total Models: {total_models}")
    print(f"  Total Authors: {total_authors}")
    print(f"  Total Tags: {total_tags}")
    print(f"  Total Downloads: {total_downloads}")
    print(f"  Total Views: {total_views}")

    # Get popular models
    print("\n2. Most popular models:")
    popular = registry.get_popular_models(limit=5)
    for i, model in enumerate(popular, 1):
        print(f"  {i}. {model['name']}")
        print(f"     Downloads: {model['downloads']}, Author: {model['author']}")

    # Get recent models
    print("\n3. Most recent models:")
    recent = registry.get_recent_models(limit=5)
    for i, model in enumerate(recent, 1):
        print(f"  {i}. {model['name']}")
        print(f"     Uploaded: {model['uploaded_at'][:10]}, Author: {model['author']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("NEURAL MARKETPLACE - COMPLETE DEMO")
    print("=" * 60)

    try:
        demo_registry()
        demo_search()
        demo_huggingface()
        demo_statistics()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nTo clean up demo files, run:")
        print("  rm -rf demo_registry/")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
