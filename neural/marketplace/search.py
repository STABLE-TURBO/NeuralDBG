"""
Semantic Search - Advanced search for Neural DSL models using embeddings and similarity.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SemanticSearch:
    """Semantic search engine for Neural DSL models."""

    def __init__(self, registry):
        """Initialize semantic search.

        Parameters
        ----------
        registry : ModelRegistry
            Model registry instance
        """
        self.registry = registry
        self.embeddings_cache = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load cached embeddings."""
        embeddings_file = self.registry.registry_dir / "embeddings.json"
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                data = json.load(f)
                self.embeddings_cache = {
                    k: np.array(v) for k, v in data.items()
                }

    def _save_embeddings(self):
        """Save embeddings to cache."""
        embeddings_file = self.registry.registry_dir / "embeddings.json"
        data = {
            k: v.tolist() for k, v in self.embeddings_cache.items()
        }
        with open(embeddings_file, 'w') as f:
            json.dump(data, f)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding using TF-IDF-like approach.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        np.ndarray
            Embedding vector
        """
        # Simple tokenization
        tokens = re.findall(r'\w+', text.lower())

        # Common ML/DL terms vocabulary
        vocab = [
            'conv', 'convolutional', 'dense', 'layer', 'neural', 'network',
            'resnet', 'vgg', 'inception', 'transformer', 'attention', 'lstm',
            'gru', 'rnn', 'classification', 'detection', 'segmentation',
            'mnist', 'cifar', 'imagenet', 'coco', 'batch', 'normalization',
            'dropout', 'activation', 'relu', 'sigmoid', 'softmax', 'pooling',
            'maxpool', 'avgpool', 'flatten', 'embedding', 'sequence',
            'image', 'text', 'nlp', 'computer', 'vision', 'deep', 'learning',
            'machine', 'model', 'training', 'inference', 'architecture'
        ]

        # Create embedding vector
        embedding = np.zeros(len(vocab) + 50)  # Extra dimensions for TF-IDF

        # Count term frequencies
        for i, term in enumerate(vocab):
            count = sum(1 for token in tokens if term in token or token in term)
            embedding[i] = count

        # Add some randomness based on text hash for uniqueness
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        embedding[len(vocab):] = np.random.randn(50) * 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _get_model_embedding(self, model_id: str) -> np.ndarray:
        """Get or create embedding for a model.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        np.ndarray
            Embedding vector
        """
        if model_id in self.embeddings_cache:
            return self.embeddings_cache[model_id]

        # Get model metadata
        model_info = self.registry.get_model_info(model_id)

        # Create text representation
        text_parts = [
            model_info.get('name', ''),
            model_info.get('description', ''),
            model_info.get('author', ''),
            ' '.join(model_info.get('tags', [])),
            model_info.get('framework', '')
        ]
        text = ' '.join(filter(None, text_parts))

        # Create embedding
        embedding = self._simple_embedding(text)
        self.embeddings_cache[model_id] = embedding
        self._save_embeddings()

        return embedding

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Parameters
        ----------
        vec1 : np.ndarray
            First vector
        vec2 : np.ndarray
            Second vector

        Returns
        -------
        float
            Cosine similarity
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for models using semantic similarity.

        Parameters
        ----------
        query : str
            Search query
        limit : int
            Maximum number of results
        filters : Dict, optional
            Additional filters (author, tags, license)

        Returns
        -------
        List[Tuple[str, float, Dict]]
            List of (model_id, similarity_score, model_info)
        """
        if filters is None:
            filters = {}

        # Get query embedding
        query_embedding = self._simple_embedding(query)

        # Get all models
        models = self.registry.list_models(
            author=filters.get('author'),
            tags=filters.get('tags')
        )

        # Filter by license if specified
        if 'license' in filters:
            models = [m for m in models if m.get('license') == filters['license']]

        # Calculate similarities
        results = []
        for model in models:
            model_id = model['id']
            model_embedding = self._get_model_embedding(model_id)
            similarity = self._cosine_similarity(query_embedding, model_embedding)
            results.append((model_id, similarity, model))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        return results[:limit]

    def search_by_architecture(
        self,
        architecture_type: str,
        limit: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for models by architecture type.

        Parameters
        ----------
        architecture_type : str
            Architecture type (e.g., 'ResNet', 'VGG', 'Transformer')
        limit : int
            Maximum number of results

        Returns
        -------
        List[Tuple[str, float, Dict]]
            List of (model_id, similarity_score, model_info)
        """
        return self.search(architecture_type, limit=limit)

    def search_by_task(
        self,
        task: str,
        limit: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for models by task.

        Parameters
        ----------
        task : str
            Task type (e.g., 'classification', 'detection', 'segmentation')
        limit : int
            Maximum number of results

        Returns
        -------
        List[Tuple[str, float, Dict]]
            List of (model_id, similarity_score, model_info)
        """
        return self.search(task, limit=limit)

    def find_similar_models(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar models to a given model.

        Parameters
        ----------
        model_id : str
            Model ID
        limit : int
            Maximum number of results

        Returns
        -------
        List[Tuple[str, float, Dict]]
            List of (model_id, similarity_score, model_info)
        """
        # Get model embedding
        model_embedding = self._get_model_embedding(model_id)

        # Get all models
        models = self.registry.list_models()

        # Calculate similarities
        results = []
        for model in models:
            other_id = model['id']
            if other_id == model_id:
                continue

            other_embedding = self._get_model_embedding(other_id)
            similarity = self._cosine_similarity(model_embedding, other_embedding)
            results.append((other_id, similarity, model))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        return results[:limit]

    def get_trending_tags(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get trending tags.

        Parameters
        ----------
        limit : int
            Maximum number of tags

        Returns
        -------
        List[Tuple[str, int]]
            List of (tag, count)
        """
        tag_counts = {}
        for tag, model_ids in self.registry.metadata["tags"].items():
            tag_counts[tag] = len(model_ids)

        # Sort by count
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:limit]

    def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        """Get autocomplete suggestions.

        Parameters
        ----------
        prefix : str
            Search prefix
        limit : int
            Maximum number of suggestions

        Returns
        -------
        List[str]
            List of suggestions
        """
        suggestions = set()

        # Search in model names
        for model in self.registry.metadata["models"].values():
            if prefix.lower() in model['name'].lower():
                suggestions.add(model['name'])

        # Search in tags
        for tag in self.registry.metadata["tags"].keys():
            if prefix.lower() in tag.lower():
                suggestions.add(tag)

        # Search in authors
        for author in self.registry.metadata["authors"].keys():
            if prefix.lower() in author.lower():
                suggestions.add(author)

        return sorted(list(suggestions))[:limit]
