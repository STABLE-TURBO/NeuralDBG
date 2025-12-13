"""
Model Registry - Storage and retrieval of Neural DSL models with versioning and licensing.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Registry for storing and managing Neural DSL models."""

    def __init__(self, registry_dir: str = "neural_marketplace_registry"):
        """Initialize the model registry.

        Parameters
        ----------
        registry_dir : str
            Directory to store the model registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)

        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.metadata_file = self.registry_dir / "registry_metadata.json"
        self.stats_file = self.registry_dir / "usage_stats.json"

        self._load_metadata()
        self._load_stats()

    def _load_metadata(self):
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "models": {},
                "tags": {},
                "authors": {}
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save registry metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _load_stats(self):
        """Load usage statistics from disk."""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
            self._save_stats()

    def _save_stats(self):
        """Save usage statistics to disk."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _generate_model_id(self, name: str, author: str) -> str:
        """Generate unique model ID."""
        base = f"{author}/{name}".lower().replace(" ", "-")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{base}-{timestamp}"

    def upload_model(
        self,
        name: str,
        author: str,
        model_path: str,
        description: str = "",
        license: str = "MIT",
        tags: Optional[List[str]] = None,
        framework: str = "neural-dsl",
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Upload a model to the registry.

        Parameters
        ----------
        name : str
            Model name
        author : str
            Model author
        model_path : str
            Path to the model file (.neural or .nr)
        description : str
            Model description
        license : str
            Model license (MIT, Apache-2.0, GPL-3.0, etc.)
        tags : List[str]
            Model tags for search
        framework : str
            Framework name
        version : str
            Model version
        metadata : Dict
            Additional metadata

        Returns
        -------
        str
            Model ID
        """
        if tags is None:
            tags = []

        if metadata is None:
            metadata = {}

        # Generate model ID
        model_id = self._generate_model_id(name, author)

        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Copy model file
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        dest_file = model_dir / model_file.name
        shutil.copy(model_file, dest_file)

        # Calculate file hash for integrity
        with open(dest_file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Create model metadata
        model_metadata = {
            "id": model_id,
            "name": name,
            "author": author,
            "description": description,
            "license": license,
            "tags": tags,
            "framework": framework,
            "version": version,
            "file": dest_file.name,
            "file_hash": file_hash,
            "uploaded_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "downloads": 0,
            "metadata": metadata
        }

        # Save model metadata
        model_metadata_file = model_dir / "metadata.json"
        with open(model_metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update registry metadata
        self.metadata["models"][model_id] = model_metadata

        # Update tags index
        for tag in tags:
            if tag not in self.metadata["tags"]:
                self.metadata["tags"][tag] = []
            if model_id not in self.metadata["tags"][tag]:
                self.metadata["tags"][tag].append(model_id)

        # Update authors index
        if author not in self.metadata["authors"]:
            self.metadata["authors"][author] = []
        if model_id not in self.metadata["authors"][author]:
            self.metadata["authors"][author].append(model_id)

        self._save_metadata()

        # Initialize stats
        self.stats[model_id] = {
            "downloads": 0,
            "views": 0,
            "last_accessed": None
        }
        self._save_stats()

        return model_id

    def download_model(self, model_id: str, output_dir: str = ".") -> str:
        """Download a model from the registry.

        Parameters
        ----------
        model_id : str
            Model ID
        output_dir : str
            Output directory

        Returns
        -------
        str
            Path to downloaded model
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_id}")

        model_metadata = self.metadata["models"][model_id]
        model_dir = self.models_dir / model_id
        source_file = model_dir / model_metadata["file"]

        if not source_file.exists():
            raise FileNotFoundError(f"Model file not found: {source_file}")

        # Copy to output directory
        output_path = Path(output_dir) / model_metadata["file"]
        shutil.copy(source_file, output_path)

        # Update download count
        model_metadata["downloads"] += 1
        self.metadata["models"][model_id] = model_metadata
        self._save_metadata()

        # Update stats
        if model_id in self.stats:
            self.stats[model_id]["downloads"] += 1
            self.stats[model_id]["last_accessed"] = datetime.now().isoformat()
        else:
            self.stats[model_id] = {
                "downloads": 1,
                "views": 0,
                "last_accessed": datetime.now().isoformat()
            }
        self._save_stats()

        return str(output_path)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        Dict
            Model metadata
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_id}")

        # Update view count
        if model_id in self.stats:
            self.stats[model_id]["views"] += 1
        else:
            self.stats[model_id] = {"downloads": 0, "views": 1, "last_accessed": None}
        self._save_stats()

        return self.metadata["models"][model_id]

    def list_models(
        self,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "uploaded_at",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List models in the registry.

        Parameters
        ----------
        author : str, optional
            Filter by author
        tags : List[str], optional
            Filter by tags (models must have all tags)
        sort_by : str
            Sort by field (uploaded_at, downloads, name)
        limit : int, optional
            Maximum number of results

        Returns
        -------
        List[Dict]
            List of model metadata
        """
        models = list(self.metadata["models"].values())

        # Filter by author
        if author:
            models = [m for m in models if m["author"] == author]

        # Filter by tags
        if tags:
            models = [m for m in models if all(tag in m["tags"] for tag in tags)]

        # Sort
        reverse = True
        if sort_by == "name":
            reverse = False
            models.sort(key=lambda m: m["name"].lower(), reverse=reverse)
        elif sort_by == "downloads":
            models.sort(key=lambda m: m["downloads"], reverse=reverse)
        elif sort_by == "uploaded_at":
            models.sort(key=lambda m: m["uploaded_at"], reverse=reverse)

        # Limit
        if limit:
            models = models[:limit]

        return models

    def update_model(
        self,
        model_id: str,
        description: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update model metadata.

        Parameters
        ----------
        model_id : str
            Model ID
        description : str, optional
            New description
        license : str, optional
            New license
        tags : List[str], optional
            New tags
        version : str, optional
            New version
        metadata : Dict, optional
            Additional metadata to update
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_id}")

        model_metadata = self.metadata["models"][model_id]

        # Update fields
        if description is not None:
            model_metadata["description"] = description
        if license is not None:
            model_metadata["license"] = license
        if version is not None:
            model_metadata["version"] = version
        if metadata is not None:
            model_metadata["metadata"].update(metadata)

        # Update tags
        if tags is not None:
            # Remove from old tags
            for tag in model_metadata["tags"]:
                if tag in self.metadata["tags"] and model_id in self.metadata["tags"][tag]:
                    self.metadata["tags"][tag].remove(model_id)

            # Add to new tags
            model_metadata["tags"] = tags
            for tag in tags:
                if tag not in self.metadata["tags"]:
                    self.metadata["tags"][tag] = []
                if model_id not in self.metadata["tags"][tag]:
                    self.metadata["tags"][tag].append(model_id)

        model_metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["models"][model_id] = model_metadata

        # Update model metadata file
        model_dir = self.models_dir / model_id
        model_metadata_file = model_dir / "metadata.json"
        with open(model_metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        self._save_metadata()

    def delete_model(self, model_id: str):
        """Delete a model from the registry.

        Parameters
        ----------
        model_id : str
            Model ID
        """
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_id}")

        model_metadata = self.metadata["models"][model_id]

        # Remove from tags
        for tag in model_metadata["tags"]:
            if tag in self.metadata["tags"] and model_id in self.metadata["tags"][tag]:
                self.metadata["tags"][tag].remove(model_id)

        # Remove from authors
        author = model_metadata["author"]
        if author in self.metadata["authors"] and model_id in self.metadata["authors"][author]:
            self.metadata["authors"][author].remove(model_id)

        # Remove from metadata
        del self.metadata["models"][model_id]
        self._save_metadata()

        # Remove stats
        if model_id in self.stats:
            del self.stats[model_id]
            self._save_stats()

        # Remove model directory
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def get_usage_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics.

        Parameters
        ----------
        model_id : str, optional
            Model ID (if None, returns all stats)

        Returns
        -------
        Dict
            Usage statistics
        """
        if model_id:
            if model_id not in self.stats:
                return {"downloads": 0, "views": 0, "last_accessed": None}
            return self.stats[model_id]
        return self.stats

    def get_popular_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular models by downloads.

        Parameters
        ----------
        limit : int
            Maximum number of results

        Returns
        -------
        List[Dict]
            List of model metadata
        """
        return self.list_models(sort_by="downloads", limit=limit)

    def get_recent_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent models.

        Parameters
        ----------
        limit : int
            Maximum number of results

        Returns
        -------
        List[Dict]
            List of model metadata
        """
        return self.list_models(sort_by="uploaded_at", limit=limit)
