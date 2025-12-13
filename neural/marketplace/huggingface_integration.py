"""
HuggingFace Hub Integration - Upload and download models from HuggingFace Hub.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional


try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file
    HF_AVAILABLE = True
except ImportError:
    HfApi = None
    hf_hub_download = None
    upload_file = None
    HF_AVAILABLE = False


class HuggingFaceIntegration:
    """Integration with HuggingFace Hub for model sharing."""

    def __init__(self, token: Optional[str] = None):
        """Initialize HuggingFace integration.

        Parameters
        ----------
        token : str, optional
            HuggingFace API token
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Install it with: pip install huggingface_hub"
            )

        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token) if self.token else None

    def upload_to_hub(
        self,
        model_path: str,
        repo_id: str,
        model_name: str,
        description: str = "",
        license: str = "mit",
        tags: Optional[List[str]] = None,
        commit_message: Optional[str] = None,
        private: bool = False
    ) -> Dict[str, str]:
        """Upload a model to HuggingFace Hub.

        Parameters
        ----------
        model_path : str
            Path to model file
        repo_id : str
            Repository ID (username/repo-name)
        model_name : str
            Model name
        description : str
            Model description
        license : str
            License identifier
        tags : List[str], optional
            Model tags
        commit_message : str, optional
            Commit message
        private : bool
            Whether to create a private repo

        Returns
        -------
        Dict
            Upload result with URL
        """
        if not self.api:
            raise ValueError("HuggingFace token required for upload")

        if tags is None:
            tags = ["neural-dsl"]
        elif "neural-dsl" not in tags:
            tags.append("neural-dsl")

        # Create repository if it doesn't exist
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
        except Exception as e:
            print(f"Repository creation note: {e}")

        # Prepare model card
        model_card = self._create_model_card(
            model_name=model_name,
            description=description,
            license=license,
            tags=tags
        )

        # Upload model card
        card_path = Path(model_path).parent / "README.md"
        with open(card_path, 'w') as f:
            f.write(model_card)

        try:
            self.api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message or f"Upload {model_name}",
            )
        finally:
            if card_path.exists():
                card_path.unlink()

        # Upload model file
        self.api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=Path(model_path).name,
            repo_id=repo_id,
            commit_message=commit_message or f"Upload {model_name}",
        )

        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "status": "success"
        }

    def download_from_hub(
        self,
        repo_id: str,
        filename: str,
        output_dir: str = ".",
        revision: str = "main"
    ) -> str:
        """Download a model from HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            Repository ID
        filename : str
            Filename to download
        output_dir : str
            Output directory
        revision : str
            Git revision (branch, tag, or commit)

        Returns
        -------
        str
            Path to downloaded file
        """
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=None,
            token=self.token
        )

        # Copy to output directory
        output_path = Path(output_dir) / filename
        shutil.copy(downloaded_path, output_path)

        return str(output_path)

    def search_hub(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for Neural DSL models on HuggingFace Hub.

        Parameters
        ----------
        query : str, optional
            Search query
        tags : List[str], optional
            Filter by tags
        limit : int
            Maximum number of results

        Returns
        -------
        List[Dict]
            List of model information
        """
        if tags is None:
            tags = ["neural-dsl"]
        elif "neural-dsl" not in tags:
            tags.append("neural-dsl")

        # Search models
        models = self.api.list_models(
            filter=tags,
            search=query,
            limit=limit,
            sort="downloads",
            direction=-1
        )

        results = []
        for model in models:
            results.append({
                "id": model.modelId,
                "author": model.author if hasattr(model, 'author') else model.modelId.split('/')[0],
                "name": model.modelId.split('/')[-1],
                "downloads": model.downloads if hasattr(model, 'downloads') else 0,
                "likes": model.likes if hasattr(model, 'likes') else 0,
                "tags": model.tags if hasattr(model, 'tags') else [],
                "updated_at": model.lastModified if hasattr(model, 'lastModified') else None,
            })

        return results

    def get_model_info(self, repo_id: str) -> Dict[str, Any]:
        """Get model information from HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            Repository ID

        Returns
        -------
        Dict
            Model information
        """
        model = self.api.model_info(repo_id, token=self.token)

        return {
            "id": model.modelId,
            "author": model.author if hasattr(model, 'author') else model.modelId.split('/')[0],
            "name": model.modelId.split('/')[-1],
            "downloads": model.downloads if hasattr(model, 'downloads') else 0,
            "likes": model.likes if hasattr(model, 'likes') else 0,
            "tags": model.tags if hasattr(model, 'tags') else [],
            "created_at": model.createdAt if hasattr(model, 'createdAt') else None,
            "updated_at": model.lastModified if hasattr(model, 'lastModified') else None,
            "private": model.private if hasattr(model, 'private') else False,
            "pipeline_tag": model.pipeline_tag if hasattr(model, 'pipeline_tag') else None,
        }

    def _create_model_card(
        self,
        model_name: str,
        description: str,
        license: str,
        tags: List[str]
    ) -> str:
        """Create a model card in Markdown format.

        Parameters
        ----------
        model_name : str
            Model name
        description : str
            Model description
        license : str
            License
        tags : List[str]
            Tags

        Returns
        -------
        str
            Model card content
        """
        card = f"""---
license: {license}
tags:
"""
        for tag in tags:
            card += f"- {tag}\n"

        card += f"""
---

# {model_name}

## Model Description

{description}

## Framework

This model was created using Neural DSL, a domain-specific language for neural network architecture design.

## Usage

To use this model with Neural DSL:

```bash
# Download the model
neural marketplace download {model_name}

# Compile and run
neural compile {model_name}.neural --backend tensorflow
neural run {model_name}_tensorflow.py
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{model_name.lower().replace(' ', '_')},
  title={{{model_name}}},
  author={{Neural DSL Community}},
  year={{2024}},
  url={{https://github.com/Lemniscate-world/Neural}}
}}
```

## License

This model is licensed under the {license.upper()} license.
"""
        return card

    def list_user_models(self, username: str) -> List[Dict[str, Any]]:
        """List all models from a user.

        Parameters
        ----------
        username : str
            Username

        Returns
        -------
        List[Dict]
            List of models
        """
        models = self.api.list_models(author=username)

        results = []
        for model in models:
            if "neural-dsl" in (model.tags if hasattr(model, 'tags') else []):
                results.append({
                    "id": model.modelId,
                    "author": username,
                    "name": model.modelId.split('/')[-1],
                    "downloads": model.downloads if hasattr(model, 'downloads') else 0,
                    "likes": model.likes if hasattr(model, 'likes') else 0,
                    "tags": model.tags if hasattr(model, 'tags') else [],
                })

        return results
