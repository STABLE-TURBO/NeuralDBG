from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .dataset_version import DatasetVersionManager
from .dvc_integration import DVCIntegration
from .feature_store import FeatureStore
from .lineage_tracker import LineageTracker
from .preprocessing_tracker import PreprocessingTracker
from .quality_validator import DataQualityValidator


def create_data_to_model_lineage(
    graph_name: str,
    dataset_version: str,
    preprocessing_pipeline: str,
    model_info: Dict[str, str],
    base_dir: Union[str, Path] = ".neural_data",
) -> str:
    tracker = LineageTracker(base_dir=base_dir)
    graph = tracker.create_graph(graph_name)
    
    model_name = model_info.get("name", "unknown_model")
    prediction_name = model_info.get("prediction", "unknown_prediction")
    
    data_node = tracker.add_data_node(
        graph_name,
        f"data_{dataset_version}",
        f"Dataset {dataset_version}",
        {"version": dataset_version},
    )
    
    preprocess_node = tracker.add_preprocessing_node(
        graph_name,
        f"preprocess_{preprocessing_pipeline}",
        f"Preprocessing {preprocessing_pipeline}",
        {"pipeline": preprocessing_pipeline},
    )
    
    model_node = tracker.add_model_node(
        graph_name,
        f"model_{model_name}",
        f"Model {model_name}",
        {"name": model_name},
    )
    
    prediction_node = tracker.add_prediction_node(
        graph_name,
        f"prediction_{prediction_name}",
        f"Predictions {prediction_name}",
        {"name": prediction_name},
    )
    
    tracker.add_edge(graph_name, data_node.node_id, preprocess_node.node_id, "input")
    tracker.add_edge(graph_name, preprocess_node.node_id, model_node.node_id, "train")
    tracker.add_edge(graph_name, model_node.node_id, prediction_node.node_id, "predict")
    
    return graph_name


def setup_data_versioning_project(
    base_dir: Union[str, Path] = ".neural_data",
    use_dvc: bool = True,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    version_manager = DatasetVersionManager(base_dir=base_dir)
    feature_store = FeatureStore(base_dir=base_dir)
    preprocessing_tracker = PreprocessingTracker(base_dir=base_dir)
    quality_validator = DataQualityValidator(base_dir=base_dir)
    lineage_tracker = LineageTracker(base_dir=base_dir)
    
    dvc_integration = None
    if use_dvc:
        dvc_integration = DVCIntegration()
        if dvc_integration.is_dvc_available():
            dvc_integration.init()
    
    return {
        "base_dir": str(base_dir),
        "version_manager": version_manager,
        "feature_store": feature_store,
        "preprocessing_tracker": preprocessing_tracker,
        "quality_validator": quality_validator,
        "lineage_tracker": lineage_tracker,
        "dvc_integration": dvc_integration,
    }


def validate_and_version_dataset(
    dataset_path: Union[str, Path],
    version: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    base_dir: Union[str, Path] = ".neural_data",
) -> Dict[str, Any]:
    version_manager = DatasetVersionManager(base_dir=base_dir)
    quality_validator = DataQualityValidator(base_dir=base_dir)
    
    options = options or {}
    validate = options.get("validate", True)
    tags = options.get("tags")
    metadata = options.get("metadata")
    
    validation_results = None
    if validate:
        try:
            import numpy as np
            
            if str(dataset_path).endswith('.npy'):
                data = np.load(dataset_path, allow_pickle=False)
            elif str(dataset_path).endswith('.npz'):
                data = np.load(dataset_path, allow_pickle=False)['data']
            else:
                try:
                    import pandas as pd
                    data = pd.read_csv(dataset_path)
                except ImportError:
                    data = np.loadtxt(dataset_path)
            
            validation_results = quality_validator.validate(data)
            
            if not all(r.passed for r in validation_results):
                failed_rules = [r.rule_name for r in validation_results if not r.passed]
                return {
                    "success": False,
                    "error": "Validation failed",
                    "failed_rules": failed_rules,
                    "validation_results": validation_results,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load or validate data: {str(e)}",
            }
    
    dataset_version = version_manager.create_version(
        dataset_path=dataset_path,
        version=version,
        metadata=metadata,
        tags=tags,
        copy_data=True,
    )
    
    return {
        "success": True,
        "version": dataset_version.version,
        "checksum": dataset_version.checksum,
        "validation_results": validation_results,
    }


def get_dataset_lineage_summary(
    graph_name: str,
    base_dir: Union[str, Path] = ".neural_data",
) -> Dict[str, Any]:
    tracker = LineageTracker(base_dir=base_dir)
    graph = tracker.get_graph(graph_name)
    
    if not graph:
        return {"error": f"Graph not found: {graph_name}"}
    
    node_types = {}
    for node in graph.nodes.values():
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
    
    data_nodes = [n for n in graph.nodes.values() if n.node_type == "data"]
    model_nodes = [n for n in graph.nodes.values() if n.node_type == "model"]
    prediction_nodes = [n for n in graph.nodes.values() if n.node_type == "prediction"]
    
    return {
        "graph_name": graph_name,
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "node_types": node_types,
        "data_nodes": [n.name for n in data_nodes],
        "model_nodes": [n.name for n in model_nodes],
        "prediction_nodes": [n.name for n in prediction_nodes],
    }


def export_data_project(
    output_dir: Union[str, Path],
    base_dir: Union[str, Path] = ".neural_data",
) -> bool:
    import shutil
    
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    
    if not base_dir.exists():
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copytree(base_dir, output_dir / base_dir.name, dirs_exist_ok=True)
    
    return True


def import_data_project(
    input_dir: Union[str, Path],
    base_dir: Union[str, Path] = ".neural_data",
) -> bool:
    import shutil
    
    input_dir = Path(input_dir)
    base_dir = Path(base_dir)
    
    if not input_dir.exists():
        return False
    
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copytree(input_dir, base_dir, dirs_exist_ok=True)
    
    return True
