"""
Base connector interface for ML platform integrations.

Provides abstract base class and common functionality for all platform connectors.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neural.exceptions import (
    CloudConnectionError,
    CloudException,
    CloudExecutionError,
    InvalidParameterError,
)


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a remote job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class ResourceConfig:
    """Configuration for compute resources."""
    instance_type: str
    gpu_enabled: bool = False
    gpu_count: int = 0
    memory_gb: Optional[int] = None
    cpu_count: Optional[int] = None
    disk_size_gb: Optional[int] = None
    max_runtime_hours: Optional[int] = None
    auto_shutdown: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result from a remote job execution."""
    job_id: str
    status: JobStatus
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    logs_url: Optional[str] = None
    duration_seconds: Optional[float] = None


class BaseConnector(ABC):
    """
    Abstract base class for ML platform connectors.
    
    All platform-specific connectors should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize the connector.
        
        Args:
            credentials: Platform-specific authentication credentials
        """
        self.credentials = credentials or {}
        self.authenticated = False
        self._session = None
        
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the platform.
        
        Returns:
            True if authentication succeeded, False otherwise
            
        Raises:
            CloudConnectionError: If authentication fails
        """
        pass
    
    @abstractmethod
    def submit_job(
        self,
        code: str,
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """
        Submit a job for execution.
        
        Args:
            code: Neural DSL code or Python code to execute
            resource_config: Resource configuration for the job
            environment: Environment variables to set
            dependencies: Python packages to install
            job_name: Optional name for the job
            
        Returns:
            Job ID for tracking the execution
            
        Raises:
            CloudExecutionError: If job submission fails
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get the status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Current status of the job
            
        Raises:
            CloudException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    def get_job_result(self, job_id: str) -> JobResult:
        """
        Get the result of a completed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job result including output, metrics, and artifacts
            
        Raises:
            CloudException: If result retrieval fails
        """
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation succeeded, False otherwise
            
        Raises:
            CloudException: If cancellation fails
        """
        pass
    
    @abstractmethod
    def list_jobs(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent jobs.
        
        Args:
            limit: Maximum number of jobs to return
            status_filter: Filter jobs by status
            
        Returns:
            List of job information dictionaries
            
        Raises:
            CloudException: If listing fails
        """
        pass
    
    @abstractmethod
    def get_logs(self, job_id: str) -> str:
        """
        Get logs for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Log output as string
            
        Raises:
            CloudException: If log retrieval fails
        """
        pass
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """
        Upload a file to the platform.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path on platform
            
        Returns:
            True if upload succeeded, False otherwise
            
        Raises:
            CloudException: If upload fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file upload"
        )
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download a file from the platform.
        
        Args:
            remote_path: Path to remote file
            local_path: Destination path locally
            
        Returns:
            True if download succeeded, False otherwise
            
        Raises:
            CloudException: If download fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file download"
        )
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """
        Deploy a model as an endpoint.
        
        Args:
            model_path: Path to model file
            endpoint_name: Name for the endpoint
            resource_config: Resource configuration for the endpoint
            
        Returns:
            Endpoint URL or identifier
            
        Raises:
            CloudException: If deployment fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support model deployment"
        )
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """
        Delete a deployed model endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            True if deletion succeeded, False otherwise
            
        Raises:
            CloudException: If deletion fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support endpoint deletion"
        )
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with resource usage information
            
        Raises:
            CloudException: If retrieval fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support resource usage monitoring"
        )
    
    def validate_credentials(self) -> bool:
        """
        Validate that credentials are properly configured.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        if not self.credentials:
            logger.warning("No credentials configured")
            return False
        return True
    
    def _save_credentials(self, filepath: Optional[str] = None) -> None:
        """
        Save credentials to a file.
        
        Args:
            filepath: Path to save credentials (default: ~/.neural/credentials.json)
        """
        if filepath is None:
            config_dir = Path.home() / ".neural"
            config_dir.mkdir(exist_ok=True)
            filepath = str(config_dir / "credentials.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.credentials, f, indent=2)
        
        logger.info(f"Credentials saved to {filepath}")
    
    def _load_credentials(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Load credentials from a file.
        
        Args:
            filepath: Path to credentials file
            
        Returns:
            Dictionary of credentials
        """
        if filepath is None:
            config_dir = Path.home() / ".neural"
            filepath = str(config_dir / "credentials.json")
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Credentials file not found: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return {}
