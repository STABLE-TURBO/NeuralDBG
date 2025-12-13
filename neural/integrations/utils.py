"""
Utility functions for platform integrations.

Provides helper functions for common tasks across platforms.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudException, FileOperationError


logger = logging.getLogger(__name__)


def load_credentials_from_file(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load credentials from a JSON file.
    
    Args:
        filepath: Path to credentials file (default: ~/.neural/credentials.json)
        
    Returns:
        Dictionary of credentials
        
    Raises:
        FileOperationError: If file cannot be read
    """
    if filepath is None:
        config_dir = Path.home() / ".neural"
        filepath = str(config_dir / "credentials.json")
    
    try:
        with open(filepath, 'r') as f:
            credentials = json.load(f)
        logger.info(f"Loaded credentials from {filepath}")
        return credentials
    except FileNotFoundError:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason="File not found"
        )
    except json.JSONDecodeError as e:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason=f"Invalid JSON: {e}"
        )
    except Exception as e:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason=str(e)
        )


def save_credentials_to_file(
    credentials: Dict[str, Any],
    filepath: Optional[str] = None
) -> None:
    """
    Save credentials to a JSON file.
    
    Args:
        credentials: Dictionary of credentials
        filepath: Path to save credentials (default: ~/.neural/credentials.json)
        
    Raises:
        FileOperationError: If file cannot be written
    """
    if filepath is None:
        config_dir = Path.home() / ".neural"
        config_dir.mkdir(exist_ok=True)
        filepath = str(config_dir / "credentials.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        os.chmod(filepath, 0o600)
        logger.info(f"Saved credentials to {filepath}")
    except Exception as e:
        raise FileOperationError(
            operation="write",
            filepath=filepath,
            reason=str(e)
        )


def load_platform_config(platform: str) -> Dict[str, Any]:
    """
    Load platform-specific configuration.
    
    Args:
        platform: Platform name
        
    Returns:
        Platform configuration dictionary
    """
    config_dir = Path.home() / ".neural" / "platforms"
    config_file = config_dir / f"{platform}.json"
    
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config for {platform}: {e}")
        return {}


def save_platform_config(platform: str, config: Dict[str, Any]) -> None:
    """
    Save platform-specific configuration.
    
    Args:
        platform: Platform name
        config: Configuration dictionary
    """
    config_dir = Path.home() / ".neural" / "platforms"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / f"{platform}.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config for {platform}")
    except Exception as e:
        logger.warning(f"Failed to save config for {platform}: {e}")


def get_environment_credentials(platform: str) -> Dict[str, Any]:
    """
    Get credentials from environment variables.
    
    Args:
        platform: Platform name
        
    Returns:
        Dictionary of credentials from environment
    """
    credentials = {}
    
    env_mapping = {
        'databricks': {
            'host': 'DATABRICKS_HOST',
            'token': 'DATABRICKS_TOKEN',
            'cluster_id': 'DATABRICKS_CLUSTER_ID',
        },
        'sagemaker': {
            'access_key_id': 'AWS_ACCESS_KEY_ID',
            'secret_access_key': 'AWS_SECRET_ACCESS_KEY',
            'region': 'AWS_DEFAULT_REGION',
            'role_arn': 'SAGEMAKER_ROLE_ARN',
        },
        'vertex_ai': {
            'project_id': 'GOOGLE_CLOUD_PROJECT',
            'location': 'GOOGLE_CLOUD_LOCATION',
            'credentials_file': 'GOOGLE_APPLICATION_CREDENTIALS',
        },
        'azure_ml': {
            'subscription_id': 'AZURE_SUBSCRIPTION_ID',
            'resource_group': 'AZURE_RESOURCE_GROUP',
            'workspace_name': 'AZURE_WORKSPACE_NAME',
        },
        'paperspace': {
            'api_key': 'PAPERSPACE_API_KEY',
            'project_id': 'PAPERSPACE_PROJECT_ID',
        },
        'runai': {
            'cluster_url': 'RUNAI_CLUSTER_URL',
            'kubeconfig': 'KUBECONFIG',
            'project': 'RUNAI_PROJECT',
        },
    }
    
    if platform not in env_mapping:
        return credentials
    
    for key, env_var in env_mapping[platform].items():
        value = os.environ.get(env_var)
        if value:
            credentials[key] = value
    
    return credentials


def validate_dsl_code(code: str) -> bool:
    """
    Validate Neural DSL code syntax.
    
    Args:
        code: Neural DSL code
        
    Returns:
        True if valid, False otherwise
    """
    if not code or not code.strip():
        return False
    
    try:
        from neural.parser.parser import parse_dsl
        parse_dsl(code)
        return True
    except Exception as e:
        logger.warning(f"DSL validation failed: {e}")
        return False


def format_job_output(result: Any) -> str:
    """
    Format job result for display.
    
    Args:
        result: Job result object
        
    Returns:
        Formatted string
    """
    from .base import JobResult
    
    if not isinstance(result, JobResult):
        return str(result)
    
    lines = [
        f"Job ID: {result.job_id}",
        f"Status: {result.status.value}",
    ]
    
    if result.duration_seconds:
        lines.append(f"Duration: {result.duration_seconds:.2f}s")
    
    if result.error:
        lines.append(f"Error: {result.error}")
    
    if result.metrics:
        lines.append("Metrics:")
        for key, value in result.metrics.items():
            lines.append(f"  {key}: {value}")
    
    if result.logs_url:
        lines.append(f"Logs: {result.logs_url}")
    
    if result.output:
        lines.append(f"\nOutput:\n{result.output[:500]}")
        if len(result.output) > 500:
            lines.append("... (truncated)")
    
    return '\n'.join(lines)


def estimate_resource_cost(
    platform: str,
    instance_type: str,
    duration_hours: float,
    gpu_enabled: bool = False
) -> Optional[float]:
    """
    Estimate cost for running a job.
    
    Args:
        platform: Platform name
        instance_type: Instance type
        duration_hours: Estimated duration in hours
        gpu_enabled: Whether GPU is used
        
    Returns:
        Estimated cost in USD or None if unknown
    """
    pricing = {
        'sagemaker': {
            'ml.t2.medium': 0.065,
            'ml.m5.large': 0.134,
            'ml.p3.2xlarge': 3.825,
        },
        'databricks': {
            'i3.xlarge': 0.312,
            'p3.2xlarge': 3.06,
        },
        'vertex_ai': {
            'n1-standard-4': 0.19,
            'n1-standard-8': 0.38,
        },
        'paperspace': {
            'P4000': 0.51,
            'P5000': 0.78,
            'V100': 2.30,
        },
    }
    
    if platform not in pricing:
        return None
    
    if instance_type not in pricing[platform]:
        return None
    
    hourly_rate = pricing[platform][instance_type]
    return hourly_rate * duration_hours


def batch_submit_jobs(
    manager: Any,
    platform: str,
    jobs: List[Dict[str, Any]]
) -> List[str]:
    """
    Submit multiple jobs in batch.
    
    Args:
        manager: PlatformManager instance
        platform: Platform name
        jobs: List of job configurations
        
    Returns:
        List of job IDs
    """
    job_ids = []
    
    for i, job_config in enumerate(jobs):
        try:
            job_id = manager.submit_job(platform=platform, **job_config)
            job_ids.append(job_id)
            logger.info(f"Submitted job {i+1}/{len(jobs)}: {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit job {i+1}: {e}")
            job_ids.append(None)
    
    return job_ids


def wait_for_job_completion(
    manager: Any,
    platform: str,
    job_id: str,
    poll_interval: int = 30,
    timeout: Optional[int] = None
) -> Any:
    """
    Wait for a job to complete.
    
    Args:
        manager: PlatformManager instance
        platform: Platform name
        job_id: Job identifier
        poll_interval: Polling interval in seconds
        timeout: Timeout in seconds
        
    Returns:
        Job result
        
    Raises:
        CloudException: If job fails or times out
    """
    import time
    from .base import JobStatus
    
    start_time = time.time()
    
    while True:
        status = manager.get_job_status(platform, job_id)
        
        if status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return manager.get_job_result(platform, job_id)
        
        if timeout and (time.time() - start_time) > timeout:
            raise CloudException(f"Job {job_id} timed out after {timeout}s")
        
        time.sleep(poll_interval)


def compare_platforms(
    jobs: List[Dict[str, Any]],
    platforms: List[str]
) -> Dict[str, Any]:
    """
    Compare cost and performance across platforms.
    
    Args:
        jobs: List of job configurations
        platforms: List of platform names
        
    Returns:
        Comparison results
    """
    comparison = {
        'platforms': platforms,
        'estimated_costs': {},
        'recommendations': []
    }
    
    for platform in platforms:
        total_cost = 0.0
        for job in jobs:
            cost = estimate_resource_cost(
                platform,
                job.get('instance_type', ''),
                job.get('duration_hours', 1.0),
                job.get('gpu_enabled', False)
            )
            if cost:
                total_cost += cost
        
        comparison['estimated_costs'][platform] = total_cost
    
    if comparison['estimated_costs']:
        cheapest = min(comparison['estimated_costs'].items(), key=lambda x: x[1])
        comparison['recommendations'].append(
            f"Most cost-effective: {cheapest[0]} (${cheapest[1]:.2f})"
        )
    
    return comparison
