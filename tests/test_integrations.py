"""
Basic tests for Neural DSL platform integrations.

These tests verify the structure and basic functionality of the integrations module.
"""

import pytest

from neural.integrations import (
    AzureMLConnector,
    BaseConnector,
    DatabricksConnector,
    JobStatus,
    PaperspaceConnector,
    PlatformManager,
    ResourceConfig,
    RunAIConnector,
    SageMakerConnector,
    VertexAIConnector,
)


class TestResourceConfig:
    """Test ResourceConfig dataclass."""
    
    def test_default_config(self):
        """Test default resource configuration."""
        config = ResourceConfig(instance_type='test-instance')
        assert config.instance_type == 'test-instance'
        assert config.gpu_enabled is False
        assert config.gpu_count == 0
        assert config.auto_shutdown is True
    
    def test_gpu_config(self):
        """Test GPU resource configuration."""
        config = ResourceConfig(
            instance_type='gpu-instance',
            gpu_enabled=True,
            gpu_count=4,
            memory_gb=64
        )
        assert config.gpu_enabled is True
        assert config.gpu_count == 4
        assert config.memory_gb == 64


class TestJobStatus:
    """Test JobStatus enumeration."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert JobStatus.PENDING.value == 'pending'
        assert JobStatus.RUNNING.value == 'running'
        assert JobStatus.SUCCEEDED.value == 'succeeded'
        assert JobStatus.FAILED.value == 'failed'
        assert JobStatus.CANCELLED.value == 'cancelled'
        assert JobStatus.UNKNOWN.value == 'unknown'


class TestBaseConnector:
    """Test BaseConnector abstract class."""
    
    def test_base_connector_abstract(self):
        """Test that BaseConnector cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseConnector()


class TestConnectorInitialization:
    """Test connector initialization."""
    
    def test_databricks_init(self):
        """Test Databricks connector initialization."""
        connector = DatabricksConnector(credentials={
            'host': 'https://test.databricks.com',
            'token': 'test-token'
        })
        assert connector.host == 'https://test.databricks.com'
        assert connector.token == 'test-token'
        assert not connector.authenticated
    
    def test_sagemaker_init(self):
        """Test SageMaker connector initialization."""
        connector = SageMakerConnector(credentials={
            'region': 'us-west-2'
        })
        assert connector.region == 'us-west-2'
        assert not connector.authenticated
    
    def test_vertex_ai_init(self):
        """Test Vertex AI connector initialization."""
        connector = VertexAIConnector(credentials={
            'project_id': 'test-project',
            'location': 'us-central1'
        })
        assert connector.project_id == 'test-project'
        assert connector.location == 'us-central1'
        assert not connector.authenticated
    
    def test_azure_ml_init(self):
        """Test Azure ML connector initialization."""
        connector = AzureMLConnector(credentials={
            'subscription_id': 'test-sub',
            'resource_group': 'test-rg',
            'workspace_name': 'test-ws'
        })
        assert connector.subscription_id == 'test-sub'
        assert connector.resource_group == 'test-rg'
        assert connector.workspace_name == 'test-ws'
        assert not connector.authenticated
    
    def test_paperspace_init(self):
        """Test Paperspace connector initialization."""
        connector = PaperspaceConnector(credentials={
            'api_key': 'test-key'
        })
        assert connector.api_key == 'test-key'
        assert not connector.authenticated
    
    def test_runai_init(self):
        """Test Run:AI connector initialization."""
        connector = RunAIConnector(credentials={
            'project': 'test-project'
        })
        assert connector.project == 'test-project'
        assert not connector.authenticated


class TestPlatformManager:
    """Test PlatformManager."""
    
    def test_manager_init(self):
        """Test manager initialization."""
        manager = PlatformManager()
        assert manager._active_platform is None
        assert len(manager._connectors) == 0
    
    def test_list_platforms(self):
        """Test listing available platforms."""
        manager = PlatformManager()
        platforms = manager.list_platforms()
        
        assert 'databricks' in platforms
        assert 'sagemaker' in platforms
        assert 'vertex_ai' in platforms
        assert 'azure_ml' in platforms
        assert 'paperspace' in platforms
        assert 'runai' in platforms
    
    def test_platform_info(self):
        """Test getting platform information."""
        manager = PlatformManager()
        info = manager.get_platform_info('databricks')
        
        assert info['name'] == 'databricks'
        assert info['configured'] is False
        assert info['active'] is False
        assert 'description' in info
    
    def test_get_all_platform_info(self):
        """Test getting all platform information."""
        manager = PlatformManager()
        all_info = manager.get_all_platform_info()
        
        assert len(all_info) == 6
        assert all(isinstance(info, dict) for info in all_info)
        assert all('name' in info for info in all_info)
    
    def test_invalid_platform(self):
        """Test handling of invalid platform."""
        from neural.exceptions import InvalidParameterError
        
        manager = PlatformManager()
        
        with pytest.raises(InvalidParameterError):
            manager.configure_platform('invalid_platform')
    
    def test_get_connector_without_config(self):
        """Test getting connector without configuration."""
        from neural.exceptions import CloudException
        
        manager = PlatformManager()
        
        with pytest.raises(CloudException):
            manager._get_connector('databricks')


class TestIntegrationImports:
    """Test that all integration components can be imported."""
    
    def test_import_base(self):
        """Test importing base components."""
        from neural.integrations.base import BaseConnector, JobResult, JobStatus, ResourceConfig
        assert BaseConnector is not None
        assert ResourceConfig is not None
        assert JobStatus is not None
        assert JobResult is not None
    
    def test_import_connectors(self):
        """Test importing all connectors."""
        from neural.integrations.databricks import DatabricksConnector
        from neural.integrations.sagemaker import SageMakerConnector
        from neural.integrations.vertex_ai import VertexAIConnector
        from neural.integrations.azure_ml import AzureMLConnector
        from neural.integrations.paperspace import PaperspaceConnector
        from neural.integrations.runai import RunAIConnector
        
        assert DatabricksConnector is not None
        assert SageMakerConnector is not None
        assert VertexAIConnector is not None
        assert AzureMLConnector is not None
        assert PaperspaceConnector is not None
        assert RunAIConnector is not None
    
    def test_import_manager(self):
        """Test importing manager."""
        from neural.integrations.manager import PlatformManager
        assert PlatformManager is not None
    
    def test_import_utils(self):
        """Test importing utilities."""
        from neural.integrations.utils import (
            format_job_output,
            get_environment_credentials,
        )
        assert format_job_output is not None
        assert get_environment_credentials is not None


class TestUtilities:
    """Test utility functions."""
    
    def test_format_job_output(self):
        """Test job output formatting."""
        from neural.integrations.base import JobResult
        from neural.integrations.utils import format_job_output
        
        result = JobResult(
            job_id='test-123',
            status=JobStatus.SUCCEEDED,
            output='Test output',
            duration_seconds=10.5
        )
        
        formatted = format_job_output(result)
        assert 'test-123' in formatted
        assert 'succeeded' in formatted
        assert '10.50s' in formatted
    
    def test_get_environment_credentials(self):
        """Test getting credentials from environment."""
        from neural.integrations.utils import get_environment_credentials
        
        creds = get_environment_credentials('databricks')
        assert isinstance(creds, dict)
    
    def test_estimate_resource_cost(self):
        """Test resource cost estimation."""
        from neural.integrations.utils import estimate_resource_cost
        
        cost = estimate_resource_cost(
            platform='sagemaker',
            instance_type='ml.t2.medium',
            duration_hours=1.0,
            gpu_enabled=False
        )
        
        assert cost is not None
        assert cost > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
