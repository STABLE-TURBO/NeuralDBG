"""
Neural DSL Platform Integrations Module.

This module provides comprehensive integration connectors for popular ML platforms
including Databricks, AWS SageMaker, Google Vertex AI, Azure ML, Paperspace Gradient,
and Run:AI.

Features
--------
- Unified API across all platforms
- Authentication and credential management
- Remote execution and resource management
- Model deployment and serving
- Experiment tracking integration

Classes
-------
BaseConnector
    Abstract base class for all platform connectors
DatabricksConnector
    Databricks notebooks and clusters integration
SageMakerConnector
    AWS SageMaker Studio integration
VertexAIConnector
    Google Vertex AI integration
AzureMLConnector
    Azure ML Studio integration
PaperspaceConnector
    Paperspace Gradient integration
RunAIConnector
    Run:AI cluster integration
PlatformManager
    Unified manager for all platform integrations

Examples
--------
>>> from neural.integrations import PlatformManager
>>> manager = PlatformManager()
>>> manager.configure_platform('databricks', host='...', token='...')
>>> job_id = manager.submit_job('databricks', code=dsl_code)
>>> status = manager.get_job_status('databricks', job_id)
"""

from .base import BaseConnector, ResourceConfig, JobStatus
from .databricks import DatabricksConnector
from .sagemaker import SageMakerConnector
from .vertex_ai import VertexAIConnector
from .azure_ml import AzureMLConnector
from .paperspace import PaperspaceConnector
from .runai import RunAIConnector
from .manager import PlatformManager

__all__ = [
    'BaseConnector',
    'ResourceConfig',
    'JobStatus',
    'DatabricksConnector',
    'SageMakerConnector',
    'VertexAIConnector',
    'AzureMLConnector',
    'PaperspaceConnector',
    'RunAIConnector',
    'PlatformManager',
]
