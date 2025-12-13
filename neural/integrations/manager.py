"""
Unified platform manager for Neural DSL integrations.

Provides a single interface to manage multiple platform connectors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from neural.exceptions import CloudException, InvalidParameterError

from .azure_ml import AzureMLConnector
from .base import BaseConnector, JobResult, JobStatus, ResourceConfig
from .databricks import DatabricksConnector
from .paperspace import PaperspaceConnector
from .runai import RunAIConnector
from .sagemaker import SageMakerConnector
from .vertex_ai import VertexAIConnector


logger = logging.getLogger(__name__)


class PlatformManager:
    """
    Unified manager for all ML platform integrations.
    
    This class provides a single interface to interact with multiple
    ML platforms through a consistent API.
    
    Examples
    --------
    >>> manager = PlatformManager()
    >>> manager.configure_platform('databricks', host='https://...', token='...')
    >>> job_id = manager.submit_job('databricks', code=dsl_code)
    >>> status = manager.get_job_status('databricks', job_id)
    """
    
    PLATFORMS: Dict[str, Type[BaseConnector]] = {
        'databricks': DatabricksConnector,
        'sagemaker': SageMakerConnector,
        'vertex_ai': VertexAIConnector,
        'azure_ml': AzureMLConnector,
        'paperspace': PaperspaceConnector,
        'runai': RunAIConnector,
    }
    
    def __init__(self):
        """Initialize the platform manager."""
        self._connectors: Dict[str, BaseConnector] = {}
        self._active_platform: Optional[str] = None
        
    def configure_platform(
        self,
        platform: str,
        credentials: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Configure a platform connector.
        
        Args:
            platform: Platform name (databricks, sagemaker, vertex_ai, etc.)
            credentials: Platform-specific credentials
            **kwargs: Additional credentials as keyword arguments
            
        Returns:
            True if configuration succeeded, False otherwise
            
        Raises:
            InvalidParameterError: If platform is not supported
            CloudException: If configuration fails
        """
        if platform not in self.PLATFORMS:
            raise InvalidParameterError(
                parameter='platform',
                value=platform,
                expected=f"One of: {', '.join(self.PLATFORMS.keys())}"
            )
        
        try:
            if credentials is None:
                credentials = {}
            credentials.update(kwargs)
            
            connector_class = self.PLATFORMS[platform]
            connector = connector_class(credentials=credentials)
            
            if connector.authenticate():
                self._connectors[platform] = connector
                self._active_platform = platform
                logger.info(f"Platform configured successfully: {platform}")
                return True
            else:
                logger.error(f"Failed to authenticate with {platform}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure {platform}: {e}")
            raise CloudException(f"Platform configuration failed: {e}")
    
    def set_active_platform(self, platform: str) -> bool:
        """
        Set the active platform for operations.
        
        Args:
            platform: Platform name
            
        Returns:
            True if successful, False otherwise
        """
        if platform not in self._connectors:
            logger.error(f"Platform not configured: {platform}")
            return False
        
        self._active_platform = platform
        logger.info(f"Active platform set to: {platform}")
        return True
    
    def submit_job(
        self,
        platform: Optional[str] = None,
        code: str = "",
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """
        Submit a job to a platform.
        
        Args:
            platform: Platform name (uses active if not specified)
            code: Neural DSL code or Python code to execute
            resource_config: Resource configuration
            environment: Environment variables
            dependencies: Python dependencies
            job_name: Job name
            
        Returns:
            Job ID
            
        Raises:
            CloudException: If job submission fails
        """
        connector = self._get_connector(platform)
        return connector.submit_job(
            code=code,
            resource_config=resource_config,
            environment=environment,
            dependencies=dependencies,
            job_name=job_name
        )
    
    def get_job_status(
        self,
        platform: Optional[str] = None,
        job_id: str = ""
    ) -> JobStatus:
        """
        Get the status of a job.
        
        Args:
            platform: Platform name (uses active if not specified)
            job_id: Job identifier
            
        Returns:
            Job status
            
        Raises:
            CloudException: If status retrieval fails
        """
        connector = self._get_connector(platform)
        return connector.get_job_status(job_id)
    
    def get_job_result(
        self,
        platform: Optional[str] = None,
        job_id: str = ""
    ) -> JobResult:
        """
        Get the result of a job.
        
        Args:
            platform: Platform name (uses active if not specified)
            job_id: Job identifier
            
        Returns:
            Job result
            
        Raises:
            CloudException: If result retrieval fails
        """
        connector = self._get_connector(platform)
        return connector.get_job_result(job_id)
    
    def cancel_job(
        self,
        platform: Optional[str] = None,
        job_id: str = ""
    ) -> bool:
        """
        Cancel a running job.
        
        Args:
            platform: Platform name (uses active if not specified)
            job_id: Job identifier
            
        Returns:
            True if cancellation succeeded, False otherwise
        """
        connector = self._get_connector(platform)
        return connector.cancel_job(job_id)
    
    def list_jobs(
        self,
        platform: Optional[str] = None,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent jobs.
        
        Args:
            platform: Platform name (uses active if not specified)
            limit: Maximum number of jobs to return
            status_filter: Filter by job status
            
        Returns:
            List of job information
            
        Raises:
            CloudException: If listing fails
        """
        connector = self._get_connector(platform)
        return connector.list_jobs(limit=limit, status_filter=status_filter)
    
    def get_logs(
        self,
        platform: Optional[str] = None,
        job_id: str = ""
    ) -> str:
        """
        Get logs for a job.
        
        Args:
            platform: Platform name (uses active if not specified)
            job_id: Job identifier
            
        Returns:
            Log output
            
        Raises:
            CloudException: If log retrieval fails
        """
        connector = self._get_connector(platform)
        return connector.get_logs(job_id)
    
    def deploy_model(
        self,
        platform: Optional[str] = None,
        model_path: str = "",
        endpoint_name: str = "",
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """
        Deploy a model as an endpoint.
        
        Args:
            platform: Platform name (uses active if not specified)
            model_path: Path to model file
            endpoint_name: Name for the endpoint
            resource_config: Resource configuration
            
        Returns:
            Endpoint URL or identifier
            
        Raises:
            CloudException: If deployment fails
        """
        connector = self._get_connector(platform)
        return connector.deploy_model(
            model_path=model_path,
            endpoint_name=endpoint_name,
            resource_config=resource_config
        )
    
    def delete_endpoint(
        self,
        platform: Optional[str] = None,
        endpoint_name: str = ""
    ) -> bool:
        """
        Delete a deployed endpoint.
        
        Args:
            platform: Platform name (uses active if not specified)
            endpoint_name: Name of the endpoint
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        connector = self._get_connector(platform)
        return connector.delete_endpoint(endpoint_name)
    
    def upload_file(
        self,
        platform: Optional[str] = None,
        local_path: str = "",
        remote_path: str = ""
    ) -> bool:
        """
        Upload a file to the platform.
        
        Args:
            platform: Platform name (uses active if not specified)
            local_path: Path to local file
            remote_path: Destination path
            
        Returns:
            True if upload succeeded, False otherwise
        """
        connector = self._get_connector(platform)
        return connector.upload_file(local_path, remote_path)
    
    def download_file(
        self,
        platform: Optional[str] = None,
        remote_path: str = "",
        local_path: str = ""
    ) -> bool:
        """
        Download a file from the platform.
        
        Args:
            platform: Platform name (uses active if not specified)
            remote_path: Path to remote file
            local_path: Destination path
            
        Returns:
            True if download succeeded, False otherwise
        """
        connector = self._get_connector(platform)
        return connector.download_file(remote_path, local_path)
    
    def get_resource_usage(
        self,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Args:
            platform: Platform name (uses active if not specified)
            
        Returns:
            Resource usage information
        """
        connector = self._get_connector(platform)
        return connector.get_resource_usage()
    
    def list_platforms(self) -> List[str]:
        """
        List available platforms.
        
        Returns:
            List of platform names
        """
        return list(self.PLATFORMS.keys())
    
    def list_configured_platforms(self) -> List[str]:
        """
        List configured platforms.
        
        Returns:
            List of configured platform names
        """
        return list(self._connectors.keys())
    
    def get_active_platform(self) -> Optional[str]:
        """
        Get the active platform name.
        
        Returns:
            Active platform name or None
        """
        return self._active_platform
    
    def _get_connector(self, platform: Optional[str] = None) -> BaseConnector:
        """
        Get a connector for the specified platform.
        
        Args:
            platform: Platform name (uses active if not specified)
            
        Returns:
            Platform connector
            
        Raises:
            CloudException: If platform is not configured
        """
        platform = platform or self._active_platform
        
        if platform is None:
            raise CloudException(
                "No platform specified and no active platform set"
            )
        
        if platform not in self._connectors:
            raise CloudException(
                f"Platform not configured: {platform}. "
                f"Call configure_platform() first."
            )
        
        return self._connectors[platform]
    
    def get_platform_info(self, platform: str) -> Dict[str, Any]:
        """
        Get information about a platform.
        
        Args:
            platform: Platform name
            
        Returns:
            Platform information
        """
        if platform not in self.PLATFORMS:
            raise InvalidParameterError(
                parameter='platform',
                value=platform,
                expected=f"One of: {', '.join(self.PLATFORMS.keys())}"
            )
        
        connector_class = self.PLATFORMS[platform]
        
        info = {
            'name': platform,
            'class': connector_class.__name__,
            'configured': platform in self._connectors,
            'active': platform == self._active_platform,
            'description': connector_class.__doc__.strip().split('\n')[0] if connector_class.__doc__ else '',
        }
        
        return info
    
    def get_all_platform_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all platforms.
        
        Returns:
            List of platform information dictionaries
        """
        return [self.get_platform_info(platform) for platform in self.PLATFORMS]
