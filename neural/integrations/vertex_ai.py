"""
Google Vertex AI platform connector for Neural DSL.

Provides integration with Vertex AI training, prediction, and workbench notebooks.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class VertexAIConnector(BaseConnector):
    """
    Connector for Google Vertex AI platform.
    
    Features:
    - Submit custom training jobs
    - Deploy models to Vertex AI endpoints
    - Execute code in Vertex AI Workbench
    - Manage compute resources
    
    Authentication:
        Uses Google Cloud credentials (project_id, location, credentials_file)
        or Application Default Credentials.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize Vertex AI connector.
        
        Args:
            credentials: Dictionary with GCP credentials
        """
        super().__init__(credentials)
        self.project_id = self.credentials.get('project_id')
        self.location = self.credentials.get('location', 'us-central1')
        self.credentials_file = self.credentials.get('credentials_file')
        self._aiplatform = None
        
    def authenticate(self) -> bool:
        """Authenticate with Google Vertex AI."""
        try:
            from google.cloud import aiplatform
            import google.auth
            
            if self.credentials_file:
                import os
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_file
            
            if not self.project_id:
                _, project = google.auth.default()
                self.project_id = project
            
            if not self.project_id:
                raise CloudConnectionError("Google Cloud project_id required")
            
            aiplatform.init(project=self.project_id, location=self.location)
            self._aiplatform = aiplatform
            
            self.authenticated = True
            logger.info(f"Successfully authenticated with Vertex AI (project: {self.project_id})")
            return True
            
        except ImportError:
            raise CloudConnectionError(
                "google-cloud-aiplatform library required. "
                "Install with: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            raise CloudConnectionError(f"Vertex AI authentication failed: {e}")
    
    def submit_job(
        self,
        code: str,
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """Submit a custom training job to Vertex AI."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            job_name = job_name or f"neural-job-{int(time.time())}"
            
            resource_config = resource_config or ResourceConfig(
                instance_type='n1-standard-4',
                gpu_enabled=False
            )
            
            container_uri = self._get_training_container(resource_config.gpu_enabled)
            
            requirements = dependencies or []
            requirements.append('neural-dsl')
            
            python_package = self._prepare_training_package(code, requirements)
            
            machine_type = resource_config.instance_type
            accelerator_type = None
            accelerator_count = 0
            
            if resource_config.gpu_enabled:
                accelerator_type = 'NVIDIA_TESLA_T4'
                accelerator_count = resource_config.gpu_count or 1
            
            job = aiplatform.CustomPythonPackageTrainingJob(
                display_name=job_name,
                python_package_gcs_uri=python_package,
                python_module_name='trainer.train',
                container_uri=container_uri,
            )
            
            job.run(
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                replica_count=1,
                sync=False
            )
            
            logger.info(f"Training job submitted: {job.resource_name}")
            return job.resource_name
            
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a Vertex AI training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            job = aiplatform.CustomJob(job_id)
            state = job.state.name
            
            status_map = {
                'JOB_STATE_PENDING': JobStatus.PENDING,
                'JOB_STATE_QUEUED': JobStatus.PENDING,
                'JOB_STATE_RUNNING': JobStatus.RUNNING,
                'JOB_STATE_SUCCEEDED': JobStatus.SUCCEEDED,
                'JOB_STATE_FAILED': JobStatus.FAILED,
                'JOB_STATE_CANCELLING': JobStatus.RUNNING,
                'JOB_STATE_CANCELLED': JobStatus.CANCELLED,
            }
            
            return status_map.get(state, JobStatus.UNKNOWN)
            
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a Vertex AI training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            job = aiplatform.CustomJob(job_id)
            status = self.get_job_status(job_id)
            
            error = None
            if status == JobStatus.FAILED and job.error:
                error = str(job.error)
            
            duration = None
            if job.start_time and job.end_time:
                duration = (job.end_time - job.start_time).total_seconds()
            
            return JobResult(
                job_id=job_id,
                status=status,
                error=error,
                logs_url=f"https://console.cloud.google.com/vertex-ai/training/{job_id}",
                duration_seconds=duration
            )
            
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running Vertex AI training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            job = aiplatform.CustomJob(job_id)
            job.cancel()
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
        """List recent Vertex AI training jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            jobs_list = aiplatform.CustomJob.list(
                filter=f"display_name:neural-*",
                order_by="create_time desc"
            )
            
            jobs = []
            for job in jobs_list[:limit]:
                job_status = self._map_state(job.state.name)
                
                if status_filter is None or job_status == status_filter:
                    jobs.append({
                        'job_id': job.resource_name,
                        'job_name': job.display_name,
                        'status': job_status,
                        'start_time': job.create_time,
                        'end_time': job.end_time,
                    })
            
            return jobs
            
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a Vertex AI training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import logging as cloud_logging
            
            logging_client = cloud_logging.Client(project=self.project_id)
            
            filter_str = f'resource.labels.job_id="{job_id}"'
            
            entries = logging_client.list_entries(filter_=filter_str, max_results=100)
            
            logs = []
            for entry in entries:
                logs.append(entry.payload)
            
            return '\n'.join(str(log) for log in logs)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve logs: {e}")
            return "Logs not available"
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """Deploy a model to a Vertex AI endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            resource_config = resource_config or ResourceConfig(
                instance_type='n1-standard-2',
                gpu_enabled=False
            )
            
            model = aiplatform.Model.upload(
                display_name=f"{endpoint_name}-model",
                artifact_uri=model_path,
                serving_container_image_uri=self._get_serving_container(
                    resource_config.gpu_enabled
                ),
            )
            
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            
            machine_type = resource_config.instance_type
            accelerator_type = None
            accelerator_count = 0
            
            if resource_config.gpu_enabled:
                accelerator_type = 'NVIDIA_TESLA_T4'
                accelerator_count = resource_config.gpu_count or 1
            
            model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                min_replica_count=1,
                max_replica_count=1,
            )
            
            logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
            return endpoint.resource_name
            
        except Exception as e:
            raise CloudException(f"Model deployment failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a Vertex AI endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            from google.cloud import aiplatform
            
            endpoint = aiplatform.Endpoint(endpoint_name)
            endpoint.delete(force=True)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def _prepare_training_package(
        self,
        code: str,
        requirements: List[str]
    ) -> str:
        """Prepare Python training package and upload to GCS."""
        import tempfile
        import tarfile
        from pathlib import Path
        from google.cloud import storage
        
        temp_dir = Path(tempfile.mkdtemp())
        package_dir = temp_dir / "trainer"
        package_dir.mkdir()
        
        (package_dir / "__init__.py").write_text("")
        
        train_script = package_dir / "train.py"
        train_script.write_text(f"""
import os
import sys

{code}
""")
        
        setup_py = temp_dir / "setup.py"
        setup_py.write_text(f"""
from setuptools import setup, find_packages

setup(
    name='neural-trainer',
    version='1.0',
    packages=find_packages(),
    install_requires={requirements},
)
""")
        
        tar_path = temp_dir / "trainer.tar.gz"
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(package_dir, arcname='trainer')
            tar.add(setup_py, arcname='setup.py')
        
        bucket_name = f"{self.project_id}-vertex-ai"
        blob_name = f"packages/trainer-{int(time.time())}.tar.gz"
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(tar_path))
        
        return f"gs://{bucket_name}/{blob_name}"
    
    def _get_training_container(self, gpu_enabled: bool) -> str:
        """Get Vertex AI training container."""
        if gpu_enabled:
            return "gcr.io/cloud-aiplatform/training/tf-gpu.2-12:latest"
        else:
            return "gcr.io/cloud-aiplatform/training/tf-cpu.2-12:latest"
    
    def _get_serving_container(self, gpu_enabled: bool) -> str:
        """Get Vertex AI serving container."""
        if gpu_enabled:
            return "gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-12:latest"
        else:
            return "gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-12:latest"
    
    def _map_state(self, state: str) -> JobStatus:
        """Map Vertex AI state to JobStatus."""
        status_map = {
            'JOB_STATE_PENDING': JobStatus.PENDING,
            'JOB_STATE_QUEUED': JobStatus.PENDING,
            'JOB_STATE_RUNNING': JobStatus.RUNNING,
            'JOB_STATE_SUCCEEDED': JobStatus.SUCCEEDED,
            'JOB_STATE_FAILED': JobStatus.FAILED,
            'JOB_STATE_CANCELLING': JobStatus.RUNNING,
            'JOB_STATE_CANCELLED': JobStatus.CANCELLED,
        }
        return status_map.get(state, JobStatus.UNKNOWN)
