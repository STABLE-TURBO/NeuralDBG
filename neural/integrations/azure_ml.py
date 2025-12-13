"""
Azure ML Studio platform connector for Neural DSL.

Provides integration with Azure ML workspaces, compute clusters, and model deployment.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class AzureMLConnector(BaseConnector):
    """
    Connector for Azure ML Studio platform.
    
    Features:
    - Submit training jobs to Azure ML compute
    - Deploy models to Azure ML endpoints
    - Execute code in Azure ML notebooks
    - Manage compute clusters
    
    Authentication:
        Uses Azure credentials (subscription_id, resource_group, workspace_name)
        and authentication via Azure CLI or service principal.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize Azure ML connector.
        
        Args:
            credentials: Dictionary with Azure ML credentials
        """
        super().__init__(credentials)
        self.subscription_id = self.credentials.get('subscription_id')
        self.resource_group = self.credentials.get('resource_group')
        self.workspace_name = self.credentials.get('workspace_name')
        self._workspace = None
        self._ml_client = None
        
    def authenticate(self) -> bool:
        """Authenticate with Azure ML."""
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            if not all([self.subscription_id, self.resource_group, self.workspace_name]):
                raise CloudConnectionError(
                    "Azure ML requires subscription_id, resource_group, and workspace_name"
                )
            
            credential = DefaultAzureCredential()
            
            self._ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            self._workspace = self._ml_client.workspaces.get(self.workspace_name)
            
            self.authenticated = True
            logger.info(f"Successfully authenticated with Azure ML workspace: {self.workspace_name}")
            return True
            
        except ImportError:
            raise CloudConnectionError(
                "azure-ai-ml library required. Install with: pip install azure-ai-ml"
            )
        except Exception as e:
            raise CloudConnectionError(f"Azure ML authentication failed: {e}")
    
    def submit_job(
        self,
        code: str,
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """Submit a training job to Azure ML."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from azure.ai.ml import command
            from azure.ai.ml.entities import Environment
            
            job_name = job_name or f"neural-job-{int(time.time())}"
            
            resource_config = resource_config or ResourceConfig(
                instance_type='Standard_D2_v2',
                gpu_enabled=False
            )
            
            requirements = dependencies or []
            requirements.append('neural-dsl')
            
            script_path = self._prepare_training_script(code)
            
            env = Environment(
                name=f"neural-env-{int(time.time())}",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file={
                    "name": "neural-env",
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.9",
                        {
                            "pip": requirements
                        }
                    ]
                }
            )
            
            job = command(
                code=script_path,
                command="python train.py",
                environment=env,
                compute=self._get_or_create_compute(resource_config),
                display_name=job_name,
                experiment_name="neural-experiments",
            )
            
            if environment:
                job.environment_variables = environment
            
            submitted_job = self._ml_client.jobs.create_or_update(job)
            
            logger.info(f"Training job submitted: {submitted_job.name}")
            return submitted_job.name
            
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of an Azure ML job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            job = self._ml_client.jobs.get(job_id)
            status = job.status
            
            status_map = {
                'NotStarted': JobStatus.PENDING,
                'Queued': JobStatus.PENDING,
                'Starting': JobStatus.PENDING,
                'Preparing': JobStatus.PENDING,
                'Running': JobStatus.RUNNING,
                'Finalizing': JobStatus.RUNNING,
                'Completed': JobStatus.SUCCEEDED,
                'Failed': JobStatus.FAILED,
                'Canceled': JobStatus.CANCELLED,
                'NotResponding': JobStatus.FAILED,
            }
            
            return status_map.get(status, JobStatus.UNKNOWN)
            
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of an Azure ML job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            job = self._ml_client.jobs.get(job_id)
            status = self.get_job_status(job_id)
            
            error = None
            if status == JobStatus.FAILED:
                error = getattr(job, 'error', {}).get('message', 'Unknown error')
            
            metrics = {}
            try:
                metrics = self._ml_client.jobs.get_metrics(job_id)
            except:
                pass
            
            duration = None
            if hasattr(job, 'creation_context'):
                creation_time = job.creation_context.created_at
                if hasattr(job, 'end_time') and job.end_time:
                    duration = (job.end_time - creation_time).total_seconds()
            
            return JobResult(
                job_id=job_id,
                status=status,
                error=error,
                metrics=metrics,
                logs_url=job.services.get('Studio', {}).get('endpoint') if hasattr(job, 'services') else None,
                duration_seconds=duration
            )
            
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running Azure ML job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            self._ml_client.jobs.cancel(job_id)
            logger.info(f"Training job cancelled: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """List recent Azure ML jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            jobs_list = self._ml_client.jobs.list(max_results=limit)
            
            jobs = []
            for job in jobs_list:
                job_status = self._map_status(job.status)
                
                if status_filter is None or job_status == status_filter:
                    jobs.append({
                        'job_id': job.name,
                        'job_name': job.display_name or job.name,
                        'status': job_status,
                        'start_time': job.creation_context.created_at if hasattr(job, 'creation_context') else None,
                        'end_time': job.end_time if hasattr(job, 'end_time') else None,
                    })
            
            return jobs
            
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for an Azure ML job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            logs = self._ml_client.jobs.download_logs(job_id, download_path="./logs")
            
            import os
            log_file = os.path.join("./logs", "user_logs", "std_log.txt")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    return f.read()
            
            return "Logs not available"
            
        except Exception as e:
            logger.warning(f"Failed to retrieve logs: {e}")
            return "Logs not available"
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """Deploy a model to an Azure ML endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from azure.ai.ml.entities import (
                ManagedOnlineDeployment,
                ManagedOnlineEndpoint,
                Model,
            )
            
            resource_config = resource_config or ResourceConfig(
                instance_type='Standard_DS2_v2',
                gpu_enabled=False
            )
            
            model = Model(
                path=model_path,
                name=f"{endpoint_name}-model",
                description="Neural DSL model"
            )
            registered_model = self._ml_client.models.create_or_update(model)
            
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="Neural DSL endpoint",
                auth_mode="key"
            )
            self._ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            deployment = ManagedOnlineDeployment(
                name="default",
                endpoint_name=endpoint_name,
                model=registered_model.id,
                instance_type=resource_config.instance_type,
                instance_count=1,
            )
            
            self._ml_client.online_deployments.begin_create_or_update(deployment).result()
            
            endpoint = self._ml_client.online_endpoints.get(endpoint_name)
            
            logger.info(f"Model deployed to endpoint: {endpoint.scoring_uri}")
            return endpoint.scoring_uri
            
        except Exception as e:
            raise CloudException(f"Model deployment failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete an Azure ML endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            self._ml_client.online_endpoints.begin_delete(endpoint_name).result()
            logger.info(f"Endpoint deleted: {endpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def _prepare_training_script(self, code: str) -> str:
        """Prepare training script directory."""
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        
        train_script = temp_dir / "train.py"
        train_script.write_text(f"""
import os
import sys

# Execute Neural DSL code
{code}
""")
        
        return str(temp_dir)
    
    def _get_or_create_compute(self, resource_config: ResourceConfig) -> str:
        """Get or create Azure ML compute cluster."""
        from azure.ai.ml.entities import AmlCompute
        
        compute_name = "neural-compute"
        
        try:
            compute = self._ml_client.compute.get(compute_name)
            logger.info(f"Using existing compute: {compute_name}")
            return compute_name
        except:
            logger.info(f"Creating new compute: {compute_name}")
            
            compute = AmlCompute(
                name=compute_name,
                type="amlcompute",
                size=resource_config.instance_type,
                min_instances=0,
                max_instances=1,
                idle_time_before_scale_down=300 if resource_config.auto_shutdown else None,
            )
            
            self._ml_client.compute.begin_create_or_update(compute).result()
            return compute_name
    
    def _map_status(self, status: str) -> JobStatus:
        """Map Azure ML status to JobStatus."""
        status_map = {
            'NotStarted': JobStatus.PENDING,
            'Queued': JobStatus.PENDING,
            'Starting': JobStatus.PENDING,
            'Preparing': JobStatus.PENDING,
            'Running': JobStatus.RUNNING,
            'Finalizing': JobStatus.RUNNING,
            'Completed': JobStatus.SUCCEEDED,
            'Failed': JobStatus.FAILED,
            'Canceled': JobStatus.CANCELLED,
            'NotResponding': JobStatus.FAILED,
        }
        return status_map.get(status, JobStatus.UNKNOWN)
