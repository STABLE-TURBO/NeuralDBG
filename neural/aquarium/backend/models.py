"""Data models for the backend bridge."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Backend(str, Enum):
    """Backend enumeration."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class ParserType(str, Enum):
    """Parser type enumeration."""
    NETWORK = "network"
    RESEARCH = "research"


class Framework(str, Enum):
    """Framework enumeration."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


class LayerInfo(BaseModel):
    """Layer information model."""
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None


class ShapeInfo(BaseModel):
    """Shape information model."""
    layer: str
    output_shape: List[Optional[int]]
    parameters: Optional[int] = None


class TraceInfo(BaseModel):
    """Execution trace information."""
    layer: str
    execution_time: float
    compute_time: float
    transfer_time: float
    flops: int
    memory: float


class IssueInfo(BaseModel):
    """Model issue information."""
    type: str
    severity: str
    message: str
    layer: Optional[str] = None


class OptimizationInfo(BaseModel):
    """Optimization suggestion information."""
    type: str
    description: str
    potential_improvement: Optional[str] = None


class JobInfo(BaseModel):
    """Job information model."""
    job_id: str
    job_name: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[str] = None
    error: Optional[str] = None


class ModelMetadata(BaseModel):
    """Model metadata."""
    network_name: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    num_layers: int = 0
    num_parameters: Optional[int] = None
    backend: Optional[str] = None


class CompilationResult(BaseModel):
    """Complete compilation result."""
    success: bool
    code: Optional[str] = None
    model_data: Optional[Dict[str, Any]] = None
    shape_history: Optional[List[ShapeInfo]] = None
    metadata: Optional[ModelMetadata] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


class JobExecutionResult(BaseModel):
    """Job execution result."""
    job_id: str
    status: JobStatus
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    duration: Optional[float] = None
