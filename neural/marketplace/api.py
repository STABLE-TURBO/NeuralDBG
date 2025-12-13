"""
Marketplace API - REST API for model marketplace operations.
"""

from __future__ import annotations

from typing import Optional


try:
    from flask import Flask, jsonify, request, send_file
    FLASK_AVAILABLE = True
except ImportError:
    Flask = None
    FLASK_AVAILABLE = False

from .huggingface_integration import HuggingFaceIntegration
from .registry import ModelRegistry
from .search import SemanticSearch

try:
    from neural.security import (
        load_security_config,
        create_basic_auth,
        create_jwt_auth,
        require_auth,
        apply_security_middleware,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


class MarketplaceAPI:
    """REST API for Neural Marketplace."""

    def __init__(
        self,
        registry_dir: str = "neural_marketplace_registry",
        hf_token: Optional[str] = None
    ):
        """Initialize marketplace API.

        Parameters
        ----------
        registry_dir : str
            Registry directory
        hf_token : str, optional
            HuggingFace API token
        """
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask is not installed. "
                "Install it with: pip install flask flask-cors"
            )

        self.registry = ModelRegistry(registry_dir)
        self.search = SemanticSearch(self.registry)

        try:
            self.hf = HuggingFaceIntegration(hf_token)
            self.hf_available = True
        except ImportError:
            self.hf = None
            self.hf_available = False

        self.app = Flask(__name__)
        
        # Load and apply security configuration
        if SECURITY_AVAILABLE:
            self.security_config = load_security_config()
            
            apply_security_middleware(
                self.app,
                cors_enabled=self.security_config.cors_enabled,
                cors_origins=self.security_config.cors_origins,
                cors_methods=self.security_config.cors_methods,
                cors_allow_headers=self.security_config.cors_allow_headers,
                cors_allow_credentials=self.security_config.cors_allow_credentials,
                rate_limit_enabled=self.security_config.rate_limit_enabled,
                rate_limit_requests=self.security_config.rate_limit_requests,
                rate_limit_window_seconds=self.security_config.rate_limit_window_seconds,
                security_headers_enabled=self.security_config.security_headers_enabled,
            )
            
            # Setup authentication if enabled
            self.auth_middleware = None
            if self.security_config.auth_enabled:
                if self.security_config.auth_type == 'jwt' and self.security_config.jwt_secret_key:
                    self.auth_middleware = create_jwt_auth(
                        self.security_config.jwt_secret_key,
                        self.security_config.jwt_algorithm,
                        self.security_config.jwt_expiration_hours
                    )
                elif self.security_config.auth_type == 'basic':
                    self.auth_middleware = create_basic_auth(
                        self.security_config.basic_auth_username,
                        self.security_config.basic_auth_password
                    )
        else:
            self.security_config = None
            self.auth_middleware = None
        
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""
        
        def _require_auth_if_enabled(f):
            if SECURITY_AVAILABLE and self.auth_middleware:
                return require_auth(self.auth_middleware)(f)
            return f

        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """List all models."""
            author = request.args.get('author')
            tags = request.args.getlist('tags')
            sort_by = request.args.get('sort_by', 'uploaded_at')
            limit = request.args.get('limit', type=int)

            models = self.registry.list_models(
                author=author,
                tags=tags if tags else None,
                sort_by=sort_by,
                limit=limit
            )

            return jsonify({
                "status": "success",
                "count": len(models),
                "models": models
            })

        @self.app.route('/api/models/<model_id>', methods=['GET'])
        def get_model(model_id):
            """Get model information."""
            try:
                model_info = self.registry.get_model_info(model_id)
                stats = self.registry.get_usage_stats(model_id)

                return jsonify({
                    "status": "success",
                    "model": model_info,
                    "stats": stats
                })
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 404

        @self.app.route('/api/models/<model_id>/download', methods=['POST'])
        def download_model(model_id):
            """Download a model."""
            try:
                output_dir = request.json.get('output_dir', '.') if request.json else '.'
                file_path = self.registry.download_model(model_id, output_dir)

                return jsonify({
                    "status": "success",
                    "message": "Model downloaded successfully",
                    "path": file_path
                })
            except (ValueError, FileNotFoundError) as e:
                return jsonify({"status": "error", "message": str(e)}), 404

        @self.app.route('/api/models/upload', methods=['POST'])
        @_require_auth_if_enabled
        def upload_model():
            """Upload a model."""
            try:
                data = request.json
                model_id = self.registry.upload_model(
                    name=data['name'],
                    author=data['author'],
                    model_path=data['model_path'],
                    description=data.get('description', ''),
                    license=data.get('license', 'MIT'),
                    tags=data.get('tags', []),
                    framework=data.get('framework', 'neural-dsl'),
                    version=data.get('version', '1.0.0'),
                    metadata=data.get('metadata', {})
                )

                return jsonify({
                    "status": "success",
                    "message": "Model uploaded successfully",
                    "model_id": model_id
                }), 201
            except (FileNotFoundError, KeyError) as e:
                return jsonify({"status": "error", "message": str(e)}), 400

        @self.app.route('/api/models/<model_id>', methods=['PUT'])
        @_require_auth_if_enabled
        def update_model(model_id):
            """Update model metadata."""
            try:
                data = request.json
                self.registry.update_model(
                    model_id=model_id,
                    description=data.get('description'),
                    license=data.get('license'),
                    tags=data.get('tags'),
                    version=data.get('version'),
                    metadata=data.get('metadata')
                )

                return jsonify({
                    "status": "success",
                    "message": "Model updated successfully"
                })
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 404

        @self.app.route('/api/models/<model_id>', methods=['DELETE'])
        @_require_auth_if_enabled
        def delete_model(model_id):
            """Delete a model."""
            try:
                self.registry.delete_model(model_id)
                return jsonify({
                    "status": "success",
                    "message": "Model deleted successfully"
                })
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 404

        @self.app.route('/api/search', methods=['GET'])
        def search_models():
            """Search for models."""
            query = request.args.get('q', '')
            limit = request.args.get('limit', 10, type=int)
            author = request.args.get('author')
            tags = request.args.getlist('tags')
            license_filter = request.args.get('license')

            filters = {}
            if author:
                filters['author'] = author
            if tags:
                filters['tags'] = tags
            if license_filter:
                filters['license'] = license_filter

            results = self.search.search(query, limit=limit, filters=filters)

            return jsonify({
                "status": "success",
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "model_id": r[0],
                        "similarity": float(r[1]),
                        "model": r[2]
                    }
                    for r in results
                ]
            })

        @self.app.route('/api/search/similar/<model_id>', methods=['GET'])
        def find_similar(model_id):
            """Find similar models."""
            limit = request.args.get('limit', 10, type=int)

            try:
                results = self.search.find_similar_models(model_id, limit=limit)

                return jsonify({
                    "status": "success",
                    "model_id": model_id,
                    "count": len(results),
                    "results": [
                        {
                            "model_id": r[0],
                            "similarity": float(r[1]),
                            "model": r[2]
                        }
                        for r in results
                    ]
                })
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 404

        @self.app.route('/api/search/autocomplete', methods=['GET'])
        def autocomplete():
            """Autocomplete suggestions."""
            prefix = request.args.get('q', '')
            limit = request.args.get('limit', 10, type=int)

            suggestions = self.search.autocomplete(prefix, limit=limit)

            return jsonify({
                "status": "success",
                "prefix": prefix,
                "suggestions": suggestions
            })

        @self.app.route('/api/tags', methods=['GET'])
        def get_tags():
            """Get trending tags."""
            limit = request.args.get('limit', 20, type=int)
            tags = self.search.get_trending_tags(limit=limit)

            return jsonify({
                "status": "success",
                "tags": [{"name": t[0], "count": t[1]} for t in tags]
            })

        @self.app.route('/api/popular', methods=['GET'])
        def get_popular():
            """Get popular models."""
            limit = request.args.get('limit', 10, type=int)
            models = self.registry.get_popular_models(limit=limit)

            return jsonify({
                "status": "success",
                "models": models
            })

        @self.app.route('/api/recent', methods=['GET'])
        def get_recent():
            """Get recent models."""
            limit = request.args.get('limit', 10, type=int)
            models = self.registry.get_recent_models(limit=limit)

            return jsonify({
                "status": "success",
                "models": models
            })

        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get marketplace statistics."""
            stats = {
                "total_models": len(self.registry.metadata["models"]),
                "total_authors": len(self.registry.metadata["authors"]),
                "total_tags": len(self.registry.metadata["tags"]),
                "total_downloads": sum(
                    s.get("downloads", 0)
                    for s in self.registry.stats.values()
                ),
                "total_views": sum(
                    s.get("views", 0)
                    for s in self.registry.stats.values()
                )
            }

            return jsonify({
                "status": "success",
                "stats": stats
            })

        # HuggingFace Hub integration routes
        if self.hf_available:
            @self.app.route('/api/hub/upload', methods=['POST'])
            def upload_to_hub():
                """Upload model to HuggingFace Hub."""
                try:
                    data = request.json
                    result = self.hf.upload_to_hub(
                        model_path=data['model_path'],
                        repo_id=data['repo_id'],
                        model_name=data['model_name'],
                        description=data.get('description', ''),
                        license=data.get('license', 'mit'),
                        tags=data.get('tags', []),
                        commit_message=data.get('commit_message'),
                        private=data.get('private', False)
                    )

                    return jsonify({
                        "status": "success",
                        "message": "Model uploaded to HuggingFace Hub",
                        "result": result
                    })
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 400

            @self.app.route('/api/hub/download', methods=['POST'])
            def download_from_hub():
                """Download model from HuggingFace Hub."""
                try:
                    data = request.json
                    file_path = self.hf.download_from_hub(
                        repo_id=data['repo_id'],
                        filename=data['filename'],
                        output_dir=data.get('output_dir', '.'),
                        revision=data.get('revision', 'main')
                    )

                    return jsonify({
                        "status": "success",
                        "message": "Model downloaded from HuggingFace Hub",
                        "path": file_path
                    })
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 400

            @self.app.route('/api/hub/search', methods=['GET'])
            def search_hub():
                """Search HuggingFace Hub."""
                query = request.args.get('q')
                tags = request.args.getlist('tags')
                limit = request.args.get('limit', 20, type=int)

                results = self.hf.search_hub(
                    query=query,
                    tags=tags if tags else None,
                    limit=limit
                )

                return jsonify({
                    "status": "success",
                    "count": len(results),
                    "results": results
                })

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the API server.

        Parameters
        ----------
        host : str
            Host address
        port : int
            Port number
        debug : bool
            Debug mode
        """
        ssl_context = None
        if SECURITY_AVAILABLE and self.security_config and self.security_config.ssl_enabled:
            if self.security_config.ssl_cert_file and self.security_config.ssl_key_file:
                ssl_context = (self.security_config.ssl_cert_file, self.security_config.ssl_key_file)
        
        self.app.run(
            host=host,
            port=port,
            debug=debug,
            ssl_context=ssl_context
        )

    def get_app(self):
        """Get Flask app instance.

        Returns
        -------
        Flask
            Flask app
        """
        return self.app
