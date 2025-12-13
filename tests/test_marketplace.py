"""
Tests for Neural Marketplace functionality.
"""

import os
from pathlib import Path
import shutil
import tempfile

import pytest

from neural.marketplace import ModelRegistry, SemanticSearch
from neural.marketplace.utils import (
    calculate_file_hash,
    compare_versions,
    format_model_size,
    parse_version,
    sanitize_model_name,
    validate_license,
    validate_model_file,
)


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(temp_dir)
    yield registry
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model_file(tmp_path):
    """Create a sample model file."""
    model_content = """
Network TestModel {
    Input: shape=(28, 28, 1)
    Conv2D: filters=32, kernel=(3,3), activation=relu
    MaxPooling2D: pool_size=(2,2)
    Flatten
    Dense: units=10, activation=softmax
    Output: loss=categorical_crossentropy
}
"""
    model_path = tmp_path / "test_model.neural"
    model_path.write_text(model_content)
    return str(model_path)


class TestModelRegistry:
    """Test model registry operations."""

    def test_upload_model(self, temp_registry, sample_model_file):
        """Test model upload."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file,
            description="A test model",
            license="MIT",
            tags=["test", "demo"],
            version="1.0.0"
        )

        assert model_id is not None
        assert "test-author/test-model" in model_id
        assert model_id in temp_registry.metadata["models"]

    def test_get_model_info(self, temp_registry, sample_model_file):
        """Test getting model information."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file,
            description="A test model",
            tags=["test"]
        )

        info = temp_registry.get_model_info(model_id)

        assert info["name"] == "Test Model"
        assert info["author"] == "Test Author"
        assert info["description"] == "A test model"
        assert "test" in info["tags"]

    def test_download_model(self, temp_registry, sample_model_file, tmp_path):
        """Test model download."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file
        )

        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        downloaded_path = temp_registry.download_model(model_id, str(output_dir))

        assert os.path.exists(downloaded_path)
        assert Path(downloaded_path).name == "test_model.neural"

    def test_list_models(self, temp_registry, sample_model_file):
        """Test listing models."""
        # Upload multiple models
        for i in range(3):
            temp_registry.upload_model(
                name=f"Model {i}",
                author="Test Author",
                model_path=sample_model_file,
                tags=["test"]
            )

        models = temp_registry.list_models()
        assert len(models) >= 3

        # Test filtering by author
        models = temp_registry.list_models(author="Test Author")
        assert len(models) >= 3

        # Test filtering by tags
        models = temp_registry.list_models(tags=["test"])
        assert len(models) >= 3

    def test_update_model(self, temp_registry, sample_model_file):
        """Test updating model metadata."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file,
            description="Original description",
            version="1.0.0"
        )

        temp_registry.update_model(
            model_id,
            description="Updated description",
            version="1.1.0",
            tags=["updated"]
        )

        info = temp_registry.get_model_info(model_id)
        assert info["description"] == "Updated description"
        assert info["version"] == "1.1.0"
        assert "updated" in info["tags"]

    def test_delete_model(self, temp_registry, sample_model_file):
        """Test deleting a model."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file
        )

        assert model_id in temp_registry.metadata["models"]

        temp_registry.delete_model(model_id)

        assert model_id not in temp_registry.metadata["models"]

    def test_usage_stats(self, temp_registry, sample_model_file, tmp_path):
        """Test usage statistics tracking."""
        model_id = temp_registry.upload_model(
            name="Test Model",
            author="Test Author",
            model_path=sample_model_file
        )

        # Get info (increments views)
        temp_registry.get_model_info(model_id)
        temp_registry.get_model_info(model_id)

        # Download (increments downloads)
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        temp_registry.download_model(model_id, str(output_dir))

        stats = temp_registry.get_usage_stats(model_id)
        assert stats["views"] == 2
        assert stats["downloads"] == 1


class TestSemanticSearch:
    """Test semantic search functionality."""

    def test_search(self, temp_registry, sample_model_file):
        """Test semantic search."""
        # Upload some test models
        temp_registry.upload_model(
            name="ResNet Classifier",
            author="Test Author",
            model_path=sample_model_file,
            description="A ResNet-based image classifier",
            tags=["classification", "resnet", "image"]
        )

        temp_registry.upload_model(
            name="LSTM Text Model",
            author="Test Author",
            model_path=sample_model_file,
            description="An LSTM model for text processing",
            tags=["nlp", "lstm", "text"]
        )

        search = SemanticSearch(temp_registry)

        # Search for classification models
        results = search.search("classification", limit=10)
        assert len(results) > 0

        # Check result structure
        model_id, similarity, model_info = results[0]
        assert isinstance(model_id, str)
        assert isinstance(similarity, float)
        assert isinstance(model_info, dict)

    def test_search_by_architecture(self, temp_registry, sample_model_file):
        """Test searching by architecture."""
        temp_registry.upload_model(
            name="ResNet Model",
            author="Test Author",
            model_path=sample_model_file,
            description="A ResNet architecture",
            tags=["resnet"]
        )

        search = SemanticSearch(temp_registry)
        results = search.search_by_architecture("resnet", limit=5)

        assert len(results) > 0

    def test_find_similar_models(self, temp_registry, sample_model_file):
        """Test finding similar models."""
        model_id1 = temp_registry.upload_model(
            name="CNN Model 1",
            author="Test Author",
            model_path=sample_model_file,
            description="A CNN for image classification",
            tags=["cnn", "classification"]
        )

        temp_registry.upload_model(
            name="CNN Model 2",
            author="Test Author",
            model_path=sample_model_file,
            description="Another CNN for image classification",
            tags=["cnn", "classification"]
        )

        search = SemanticSearch(temp_registry)
        similar = search.find_similar_models(model_id1, limit=5)

        assert len(similar) > 0

    def test_trending_tags(self, temp_registry, sample_model_file):
        """Test getting trending tags."""
        # Upload models with various tags
        for i in range(3):
            temp_registry.upload_model(
                name=f"Model {i}",
                author="Test Author",
                model_path=sample_model_file,
                tags=["test", "classification"]
            )

        search = SemanticSearch(temp_registry)
        tags = search.get_trending_tags(limit=10)

        assert len(tags) > 0
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in tags)

    def test_autocomplete(self, temp_registry, sample_model_file):
        """Test autocomplete."""
        temp_registry.upload_model(
            name="Classification Model",
            author="Test Author",
            model_path=sample_model_file,
            tags=["classification"]
        )

        search = SemanticSearch(temp_registry)
        suggestions = search.autocomplete("class", limit=10)

        assert len(suggestions) > 0


class TestUtils:
    """Test utility functions."""

    def test_calculate_file_hash(self, sample_model_file):
        """Test file hash calculation."""
        hash1 = calculate_file_hash(sample_model_file)
        hash2 = calculate_file_hash(sample_model_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_validate_license(self):
        """Test license validation."""
        assert validate_license("MIT") is True
        assert validate_license("Apache-2.0") is True
        assert validate_license("GPL-3.0") is True
        assert validate_license("Invalid-License") is False

    def test_format_model_size(self):
        """Test model size formatting."""
        assert "1.00 KB" in format_model_size(1024)
        assert "1.00 MB" in format_model_size(1024 * 1024)
        assert "1.00 GB" in format_model_size(1024 * 1024 * 1024)

    def test_parse_version(self):
        """Test version parsing."""
        assert parse_version("1.2.3") == (1, 2, 3)
        assert parse_version("2.0.0") == (2, 0, 0)
        assert parse_version("0.1.0") == (0, 1, 0)

    def test_compare_versions(self):
        """Test version comparison."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        assert sanitize_model_name("My Model!") == "my_model_"
        assert sanitize_model_name("Test@Model#123") == "test_model_123"

    def test_validate_model_file(self, sample_model_file):
        """Test model file validation."""
        is_valid, error = validate_model_file(sample_model_file)
        assert is_valid is True
        assert error is None

        # Test invalid file
        is_valid, error = validate_model_file("nonexistent.neural")
        assert is_valid is False
        assert error is not None


class TestIntegration:
    """Integration tests for marketplace."""

    def test_full_workflow(self, temp_registry, sample_model_file, tmp_path):
        """Test complete upload-search-download workflow."""
        # Upload a model
        model_id = temp_registry.upload_model(
            name="Integration Test Model",
            author="Test Author",
            model_path=sample_model_file,
            description="A model for integration testing",
            tags=["integration", "test"],
            license="MIT",
            version="1.0.0"
        )

        # Search for the model
        search = SemanticSearch(temp_registry)
        results = search.search("integration test", limit=10)

        assert len(results) > 0
        found_model = None
        for mid, similarity, model in results:
            if mid == model_id:
                found_model = model
                break

        assert found_model is not None
        assert found_model["name"] == "Integration Test Model"

        # Download the model
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        downloaded_path = temp_registry.download_model(model_id, str(output_dir))

        assert os.path.exists(downloaded_path)

        # Verify stats
        stats = temp_registry.get_usage_stats(model_id)
        assert stats["downloads"] >= 1

    def test_multiple_versions(self, temp_registry, sample_model_file):
        """Test handling multiple versions of a model."""
        # Upload version 1.0.0
        model_id_v1 = temp_registry.upload_model(
            name="Versioned Model",
            author="Test Author",
            model_path=sample_model_file,
            version="1.0.0"
        )

        # Upload version 2.0.0
        model_id_v2 = temp_registry.upload_model(
            name="Versioned Model",
            author="Test Author",
            model_path=sample_model_file,
            version="2.0.0"
        )

        # Both versions should exist
        info_v1 = temp_registry.get_model_info(model_id_v1)
        info_v2 = temp_registry.get_model_info(model_id_v2)

        assert info_v1["version"] == "1.0.0"
        assert info_v2["version"] == "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
