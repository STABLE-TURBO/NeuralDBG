from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class PreprocessingStep:
    """
    Represents a single step in a preprocessing pipeline.
    
    Attributes:
        name: Step name
        function: Callable function
        params: Function parameters
        description: Step description
    """
    def __init__(
        self,
        name: str,
        function: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.function = function
        self.params = params or {}
        self.description = description or ""
        self.applied_at: Optional[str] = None
        self.input_checksum: Optional[str] = None
        self.output_checksum: Optional[str] = None

    def apply(self, data: Any) -> Any:
        if self.function is None:
            raise ValueError(f"No function defined for step: {self.name}")
        
        self.applied_at = datetime.now().isoformat()
        result = self.function(data, **self.params)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "description": self.description,
            "applied_at": self.applied_at,
            "input_checksum": self.input_checksum,
            "output_checksum": self.output_checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingStep:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.function = None
        obj.params = data.get("params", {})
        obj.description = data.get("description", "")
        obj.applied_at = data.get("applied_at")
        obj.input_checksum = data.get("input_checksum")
        obj.output_checksum = data.get("output_checksum")
        return obj


class PreprocessingPipeline:
    """
    Represents a sequence of preprocessing steps.
    
    Attributes:
        name: Pipeline name
        steps: List of steps
        description: Description
    """
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or ""
        self.steps: List[PreprocessingStep] = []
        self.created_at = datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}

    def add_step(
        self,
        name: str,
        function: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> PreprocessingStep:
        step = PreprocessingStep(name, function, params, description)
        self.steps.append(step)
        return step

    def remove_step(self, name: str) -> bool:
        for i, step in enumerate(self.steps):
            if step.name == name:
                self.steps.pop(i)
                return True
        return False

    def apply(self, data: Any) -> Any:
        result = data
        for step in self.steps:
            result = step.apply(result)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingPipeline:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.description = data.get("description", "")
        obj.steps = [PreprocessingStep.from_dict(s) for s in data.get("steps", [])]
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        obj.metadata = data.get("metadata", {})
        return obj


class PreprocessingTracker:
    """
    Tracks preprocessing pipelines and their executions.
    """
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self.base_dir = Path(base_dir)
        self.pipelines_dir = self.base_dir / "preprocessing_pipelines"
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.pipelines: Dict[str, PreprocessingPipeline] = {}
        self._load_pipelines()

    def _load_pipelines(self):
        for pipeline_file in self.pipelines_dir.glob("*.json"):
            with open(pipeline_file, "r") as f:
                data = json.load(f)
                pipeline = PreprocessingPipeline.from_dict(data)
                self.pipelines[pipeline.name] = pipeline

    def _save_pipeline(self, pipeline: PreprocessingPipeline):
        pipeline_file = self.pipelines_dir / f"{pipeline.name}.json"
        with open(pipeline_file, "w") as f:
            json.dump(pipeline.to_dict(), f, indent=2)

    def create_pipeline(
        self, name: str, description: Optional[str] = None
    ) -> PreprocessingPipeline:
        if name in self.pipelines:
            raise ValueError(f"Pipeline already exists: {name}")
        
        pipeline = PreprocessingPipeline(name, description)
        self.pipelines[name] = pipeline
        self._save_pipeline(pipeline)
        return pipeline

    def get_pipeline(self, name: str) -> Optional[PreprocessingPipeline]:
        return self.pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        return list(self.pipelines.keys())

    def delete_pipeline(self, name: str) -> bool:
        if name not in self.pipelines:
            return False
        
        pipeline_file = self.pipelines_dir / f"{name}.json"
        if pipeline_file.exists():
            pipeline_file.unlink()
        
        del self.pipelines[name]
        return True

    def update_pipeline(self, pipeline: PreprocessingPipeline):
        self.pipelines[pipeline.name] = pipeline
        self._save_pipeline(pipeline)

    def get_pipeline_history(self, name: str) -> List[Dict[str, Any]]:
        pipeline = self.get_pipeline(name)
        if not pipeline:
            return []
        
        history = []
        for step in pipeline.steps:
            if step.applied_at:
                history.append({
                    "step_name": step.name,
                    "applied_at": step.applied_at,
                    "params": step.params,
                    "input_checksum": step.input_checksum,
                    "output_checksum": step.output_checksum,
                })
        
        return history

    def compare_pipelines(
        self, name1: str, name2: str
    ) -> Dict[str, Any]:
        pipeline1 = self.get_pipeline(name1)
        pipeline2 = self.get_pipeline(name2)
        
        if not pipeline1 or not pipeline2:
            raise ValueError("One or both pipelines not found")
        
        steps1 = {s.name: s for s in pipeline1.steps}
        steps2 = {s.name: s for s in pipeline2.steps}
        
        all_steps = set(steps1.keys()) | set(steps2.keys())
        
        differences = []
        for step_name in all_steps:
            if step_name not in steps1:
                differences.append({
                    "step": step_name,
                    "status": "only_in_pipeline2",
                    "params": steps2[step_name].params,
                })
            elif step_name not in steps2:
                differences.append({
                    "step": step_name,
                    "status": "only_in_pipeline1",
                    "params": steps1[step_name].params,
                })
            elif steps1[step_name].params != steps2[step_name].params:
                differences.append({
                    "step": step_name,
                    "status": "different_params",
                    "params1": steps1[step_name].params,
                    "params2": steps2[step_name].params,
                })
        
        return {
            "pipeline1": name1,
            "pipeline2": name2,
            "differences": differences,
            "same_steps": len(all_steps) - len(differences),
            "total_steps1": len(pipeline1.steps),
            "total_steps2": len(pipeline2.steps),
        }

    def export_pipeline(self, name: str, output_path: Union[str, Path]) -> bool:
        pipeline = self.get_pipeline(name)
        if not pipeline:
            return False
        
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(pipeline.to_dict(), f, indent=2)
        
        return True

    def import_pipeline(self, input_path: Union[str, Path]) -> PreprocessingPipeline:
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {input_path}")
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        pipeline = PreprocessingPipeline.from_dict(data)
        self.pipelines[pipeline.name] = pipeline
        self._save_pipeline(pipeline)
        
        return pipeline
