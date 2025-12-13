"""
Databricks platform connector for Neural DSL.

Provides integration with Databricks notebooks, clusters, and jobs API.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class DatabricksConnector(BaseConnector):
    """
    Connector for Databricks platform.
    
    Features:
    - Submit jobs to Databricks clusters
    - Create and manage notebooks
    - Execute code on existing clusters
    - Deploy models to Databricks Model Serving
    
    Authentication:
        Requires 'host' and 'token' in credentials dictionary.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize Databricks connector.
        
        Args:
            credentials: Dictionary with 'host' and 'token' keys
        """
        super().__init__(credentials)
        self.host = self.credentials.get('host', '')
        self.token = self.credentials.get('token', '')
        self.cluster_id = self.credentials.get('cluster_id')
        
    def authenticate(self) -> bool:
        """Authenticate with Databricks."""
        if not self.host or not self.token:
            raise CloudConnectionError(
                "Databricks authentication requires 'host' and 'token'"
            )
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(
                f"{self.host}/api/2.0/clusters/list",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.authenticated = True
                logger.info("Successfully authenticated with Databricks")
                return True
            else:
                raise CloudConnectionError(
                    f"Databricks authentication failed: {response.text}"
                )
        except ImportError:
            raise CloudConnectionError(
                "requests library required. Install with: pip install requests"
            )
        except Exception as e:
            raise CloudConnectionError(f"Authentication failed: {e}")
    
    def submit_job(
        self,
        code: str,
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """Submit a job to Databricks."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            notebook_task = self._create_notebook(code, job_name or "neural-job")
            
            cluster_spec = self._build_cluster_spec(resource_config)
            
            job_config = {
                "run_name": job_name or f"neural-job-{int(time.time())}",
                "notebook_task": {
                    "notebook_path": notebook_task["path"],
                },
                "new_cluster": cluster_spec if not self.cluster_id else None,
                "existing_cluster_id": self.cluster_id,
            }
            
            if dependencies:
                job_config["libraries"] = [
                    {"pypi": {"package": dep}} for dep in dependencies
                ]
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.post(
                f"{self.host}/api/2.1/jobs/runs/submit",
                headers=headers,
                json=job_config,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                job_id = str(result.get('run_id'))
                logger.info(f"Job submitted successfully: {job_id}")
                return job_id
            else:
                raise CloudExecutionError(
                    f"Failed to submit job: {response.text}"
                )
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a Databricks job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(
                f"{self.host}/api/2.1/jobs/runs/get",
                headers=headers,
                params={'run_id': job_id},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                state = result.get('state', {})
                life_cycle_state = state.get('life_cycle_state', 'UNKNOWN')
                
                status_map = {
                    'PENDING': JobStatus.PENDING,
                    'RUNNING': JobStatus.RUNNING,
                    'TERMINATING': JobStatus.RUNNING,
                    'TERMINATED': JobStatus.SUCCEEDED,
                    'SKIPPED': JobStatus.CANCELLED,
                    'INTERNAL_ERROR': JobStatus.FAILED,
                }
                
                if life_cycle_state == 'TERMINATED':
                    result_state = state.get('result_state', 'SUCCESS')
                    if result_state != 'SUCCESS':
                        return JobStatus.FAILED
                
                return status_map.get(life_cycle_state, JobStatus.UNKNOWN)
            else:
                raise CloudException(f"Failed to get job status: {response.text}")
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a Databricks job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(
                f"{self.host}/api/2.1/jobs/runs/get",
                headers=headers,
                params={'run_id': job_id},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                state = result.get('state', {})
                
                output = None
                response_output = requests.get(
                    f"{self.host}/api/2.1/jobs/runs/get-output",
                    headers=headers,
                    params={'run_id': job_id},
                    timeout=30
                )
                if response_output.status_code == 200:
                    output_data = response_output.json()
                    output = output_data.get('notebook_output', {}).get('result')
                
                return JobResult(
                    job_id=job_id,
                    status=self.get_job_status(job_id),
                    output=output,
                    error=state.get('state_message'),
                    logs_url=result.get('run_page_url'),
                    duration_seconds=result.get('execution_duration', 0) / 1000.0
                )
            else:
                raise CloudException(f"Failed to get job result: {response.text}")
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running Databricks job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.post(
                f"{self.host}/api/2.1/jobs/runs/cancel",
                headers=headers,
                json={'run_id': job_id},
                timeout=30
            )
            
            return response.status_code in [200, 201]
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """List recent Databricks jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(
                f"{self.host}/api/2.1/jobs/runs/list",
                headers=headers,
                params={'limit': limit},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                runs = result.get('runs', [])
                
                jobs = []
                for run in runs:
                    job_info = {
                        'job_id': str(run.get('run_id')),
                        'job_name': run.get('run_name', ''),
                        'status': self._map_status(run.get('state', {})),
                        'start_time': run.get('start_time'),
                        'end_time': run.get('end_time'),
                    }
                    
                    if status_filter is None or job_info['status'] == status_filter:
                        jobs.append(job_info)
                
                return jobs
            else:
                raise CloudException(f"Failed to list jobs: {response.text}")
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a Databricks job."""
        result = self.get_job_result(job_id)
        return result.output or "No logs available"
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """Deploy a model to Databricks Model Serving."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            endpoint_config = {
                "name": endpoint_name,
                "config": {
                    "served_models": [
                        {
                            "model_name": model_path,
                            "model_version": "1",
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True
                        }
                    ]
                }
            }
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.post(
                f"{self.host}/api/2.0/serving-endpoints",
                headers=headers,
                json=endpoint_config,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                endpoint_url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"
                logger.info(f"Model deployed to: {endpoint_url}")
                return endpoint_url
            else:
                raise CloudException(f"Model deployment failed: {response.text}")
        except Exception as e:
            raise CloudException(f"Deployment failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a Databricks model serving endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.delete(
                f"{self.host}/api/2.0/serving-endpoints/{endpoint_name}",
                headers=headers,
                timeout=30
            )
            
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def _create_notebook(self, code: str, name: str) -> Dict[str, Any]:
        """Create a notebook with the given code."""
        import base64
        import requests
        
        notebook_content = f"""# Databricks notebook source
# MAGIC %pip install neural-dsl

# COMMAND ----------

{code}
"""
        
        encoded_content = base64.b64encode(notebook_content.encode()).decode()
        
        notebook_path = f"/Users/{self.credentials.get('username', 'neural')}/{name}"
        
        headers = {'Authorization': f'Bearer {self.token}'}
        response = requests.post(
            f"{self.host}/api/2.0/workspace/import",
            headers=headers,
            json={
                "path": notebook_path,
                "format": "SOURCE",
                "language": "PYTHON",
                "content": encoded_content,
                "overwrite": True
            },
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            return {"path": notebook_path}
        else:
            raise CloudException(f"Failed to create notebook: {response.text}")
    
    def _build_cluster_spec(self, resource_config: Optional[ResourceConfig]) -> Dict[str, Any]:
        """Build Databricks cluster specification."""
        if resource_config is None:
            resource_config = ResourceConfig(
                instance_type="i3.xlarge",
                gpu_enabled=False
            )
        
        spec = {
            "spark_version": "12.2.x-scala2.12",
            "node_type_id": resource_config.instance_type,
            "num_workers": 1,
            "autotermination_minutes": 30 if resource_config.auto_shutdown else 0,
        }
        
        if resource_config.gpu_enabled:
            spec["spark_conf"] = {
                "spark.databricks.delta.preview.enabled": "true"
            }
        
        return spec
    
    def _map_status(self, state: Dict[str, Any]) -> JobStatus:
        """Map Databricks state to JobStatus."""
        life_cycle_state = state.get('life_cycle_state', 'UNKNOWN')
        
        status_map = {
            'PENDING': JobStatus.PENDING,
            'RUNNING': JobStatus.RUNNING,
            'TERMINATING': JobStatus.RUNNING,
            'TERMINATED': JobStatus.SUCCEEDED,
            'SKIPPED': JobStatus.CANCELLED,
            'INTERNAL_ERROR': JobStatus.FAILED,
        }
        
        if life_cycle_state == 'TERMINATED':
            result_state = state.get('result_state', 'SUCCESS')
            if result_state != 'SUCCESS':
                return JobStatus.FAILED
        
        return status_map.get(life_cycle_state, JobStatus.UNKNOWN)
