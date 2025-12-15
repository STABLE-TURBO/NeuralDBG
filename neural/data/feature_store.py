from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Feature:
    """
    Represents a single feature in the feature store.
    
    Attributes:
        name: Feature name
        dtype: Data type of the feature
        description: Optional description
        metadata: Optional metadata dictionary
    """
    def __init__(
        self,
        name: str,
        dtype: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description or ""
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Feature:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.dtype = data["dtype"]
        obj.description = data.get("description", "")
        obj.metadata = data.get("metadata", {})
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        obj.updated_at = data.get("updated_at", obj.created_at)
        return obj


class FeatureGroup:
    """
    Represents a logical grouping of features.
    
    Attributes:
        name: Group name
        features: List of Feature objects
        description: Optional description
    """
    def __init__(
        self,
        name: str,
        features: Optional[List[Feature]] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.features = features or []
        self.description = description or ""
        self.created_at = datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}

    def add_feature(self, feature: Feature):
        if any(f.name == feature.name for f in self.features):
            raise ValueError(f"Feature already exists: {feature.name}")
        self.features.append(feature)

    def remove_feature(self, name: str) -> bool:
        for i, feature in enumerate(self.features):
            if feature.name == name:
                self.features.pop(i)
                return True
        return False

    def get_feature(self, name: str) -> Optional[Feature]:
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "features": [f.to_dict() for f in self.features],
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureGroup:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.features = [Feature.from_dict(f) for f in data.get("features", [])]
        obj.description = data.get("description", "")
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        obj.metadata = data.get("metadata", {})
        return obj


class FeatureStore:
    """
    Manages feature groups and features storage.
    
    Handles serialization and persistence of feature metadata.
    """
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self.base_dir = Path(base_dir)
        self.features_dir = self.base_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self._load_feature_groups()

    def _load_feature_groups(self):
        for group_file in self.features_dir.glob("*.json"):
            with open(group_file, "r") as f:
                data = json.load(f)
                group = FeatureGroup.from_dict(data)
                self.feature_groups[group.name] = group

    def _save_feature_group(self, group: FeatureGroup):
        group_file = self.features_dir / f"{group.name}.json"
        with open(group_file, "w") as f:
            json.dump(group.to_dict(), f, indent=2)

    def create_feature_group(
        self, name: str, description: Optional[str] = None
    ) -> FeatureGroup:
        if name in self.feature_groups:
            raise ValueError(f"Feature group already exists: {name}")
        
        group = FeatureGroup(name, description=description)
        self.feature_groups[name] = group
        self._save_feature_group(group)
        return group

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        return self.feature_groups.get(name)

    def list_feature_groups(self) -> List[str]:
        return list(self.feature_groups.keys())

    def delete_feature_group(self, name: str) -> bool:
        if name not in self.feature_groups:
            return False
        
        group_file = self.features_dir / f"{name}.json"
        if group_file.exists():
            group_file.unlink()
        
        del self.feature_groups[name]
        return True

    def add_feature(
        self,
        group_name: str,
        feature_name_or_obj: Union[str, Feature],
        dtype: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feature:
        """
        Add a feature to a group.
        
        Args:
            group_name: Name of the feature group
            feature_name_or_obj: Either a Feature object or feature name string
            dtype: Data type (required if feature_name_or_obj is a string)
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The added Feature object
        """
        group = self.get_feature_group(group_name)
        if not group:
            raise ValueError(f"Feature group not found: {group_name}")
        
        # Support both Feature object and individual parameters
        if isinstance(feature_name_or_obj, Feature):
            feature = feature_name_or_obj
        else:
            # Create Feature from individual parameters
            if dtype is None:
                raise ValueError("dtype is required when providing feature name as string")
            feature = Feature(
                name=feature_name_or_obj,
                dtype=dtype,
                description=description,
                metadata=metadata
            )
        
        group.add_feature(feature)
        self._save_feature_group(group)
        return feature

    def remove_feature(self, group_name: str, feature_name: str) -> bool:
        group = self.get_feature_group(group_name)
        if not group:
            return False
        
        result = group.remove_feature(feature_name)
        if result:
            self._save_feature_group(group)
        return result

    def get_feature(
        self, group_name: str, feature_name: str
    ) -> Optional[Feature]:
        group = self.get_feature_group(group_name)
        if not group:
            return None
        return group.get_feature(feature_name)

    def search_features(
        self,
        query: str,
        group_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()
        
        groups = (
            [self.feature_groups[group_name]]
            if group_name and group_name in self.feature_groups
            else self.feature_groups.values()
        )
        
        for group in groups:
            for feature in group.features:
                if (
                    query_lower in feature.name.lower()
                    or query_lower in feature.description.lower()
                ):
                    results.append({
                        "group": group.name,
                        "feature": feature.name,
                        "dtype": feature.dtype,
                        "description": feature.description,
                    })
        
        return results

    def get_feature_statistics(self) -> Dict[str, Any]:
        total_features = sum(
            len(group.features) for group in self.feature_groups.values()
        )
        
        dtype_counts: Dict[str, int] = {}
        for group in self.feature_groups.values():
            for feature in group.features:
                dtype_counts[feature.dtype] = dtype_counts.get(feature.dtype, 0) + 1
        
        return {
            "total_groups": len(self.feature_groups),
            "total_features": total_features,
            "dtype_distribution": dtype_counts,
            "groups": {
                name: len(group.features)
                for name, group in self.feature_groups.items()
            },
        }

    def export_feature_group(
        self, name: str, output_path: Union[str, Path]
    ) -> bool:
        group = self.get_feature_group(name)
        if not group:
            return False
        
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(group.to_dict(), f, indent=2)
        
        return True

    def import_feature_group(
        self, input_path: Union[str, Path]
    ) -> FeatureGroup:
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Feature group file not found: {input_path}")
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        group = FeatureGroup.from_dict(data)
        self.feature_groups[group.name] = group
        self._save_feature_group(group)
        
        return group

    def update_feature_group(self, group: FeatureGroup):
        self.feature_groups[group.name] = group
        self._save_feature_group(group)
