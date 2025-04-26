"""
Neural DSL Cloud Integration Module

This module provides tools and utilities for running Neural DSL in cloud environments
like Kaggle, Google Colab, and AWS SageMaker.
"""

from .cloud_execution import CloudExecutor
from .remote_connection import RemoteConnection

# Import SageMaker integration if running in SageMaker environment
import os
if 'SM_MODEL_DIR' in os.environ:
    from .sagemaker_integration import SageMakerHandler, sagemaker_entry_point

__all__ = ['CloudExecutor', 'RemoteConnection']
if 'SM_MODEL_DIR' in os.environ:
    __all__.extend(['SageMakerHandler', 'sagemaker_entry_point'])
