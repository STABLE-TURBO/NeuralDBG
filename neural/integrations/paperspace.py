"""
Paperspace Gradient platform connector for Neural DSL.

Provides integration with Paperspace Gradient notebooks, jobs, and deployments.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class PaperspaceConnector(BaseConnector):
    """
    Connector for Paperspace Gradient platform.
    
    Features:
    - Submit jobs to Gradient compute
    - Create and manage notebooks
    - Deploy models to Gradient deployments
    - Manage compute resources
    
    Authentication:
        Requires 'api_key' in credentials dictionary.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize Paperspace connector.
        
        Args:
            credentials: Dictionary with 'api_key' key
        """
        super().__init__(credentials)
        self.api_key = self.credentials.get('api_key', '')
        self.base_url = 'https://api.paperspace.io'
        self.project_id = self.credentials.get('project_id')
        
    def authenticate(self) -> bool:
        """Authenticate with Paperspace Gradient."""
        if not self.api_key:
            raise CloudConnectionError(
                "Paperspace authentication requires 'api_key'"
            )
        
        try:
            import requests
            
            headers = {'X-API-Key': self.api_key}
            response = requests.get(
                f"{self.base_url}/users/getUser",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.authenticated = True
                logger.info("Successfully authenticated with Paperspace Gradient")
                return True
            else:
                raise CloudConnectionError(
                    f"Paperspace authentication failed: {response.text}"
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
        """Submit a job to Paperspace Gradient."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            resource_config = resource_config or ResourceConfig(
                instance_type='P4000',
                gpu_enabled=True
            )
            
            job_config = {
                "name": job_name or f"neural-job-{int(time.time())}",
                "machineType": resource_config.instance_type,
                "container": "tensorflow/tensorflow:latest-gpu" if resource_config.gpu_enabled else "tensorflow/tensorflow:latest",
                "command": self._build_command(code, dependencies),
            }
            
            if self.project_id:
                job_config["projectId"] = self.project_id
            
            if environment:
                job_config["environment"] = environment
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/jobs/createJob",
                headers=headers,
                json=job_config,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                job_id = result.get('id') or result.get('jobId')
                logger.info(f"Job submitted successfully: {job_id}")
                return job_id
            else:
                raise CloudExecutionError(
                    f"Failed to submit job: {response.text}"
                )
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a Paperspace job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'X-API-Key': self.api_key}
            response = requests.get(
                f"{self.base_url}/jobs/getJob",
                headers=headers,
                params={'jobId': job_id},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                state = result.get('state', 'Unknown')
                
                status_map = {
                    'Pending': JobStatus.PENDING,
                    'Running': JobStatus.RUNNING,
                    'Stopped': JobStatus.SUCCEEDED,
                    'Error': JobStatus.FAILED,
                    'Failed': JobStatus.FAILED,
                    'Cancelled': JobStatus.CANCELLED,
                }
                
                return status_map.get(state, JobStatus.UNKNOWN)
            else:
                raise CloudException(f"Failed to get job status: {response.text}")
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a Paperspace job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'X-API-Key': self.api_key}
            response = requests.get(
                f"{self.base_url}/jobs/getJob",
                headers=headers,
                params={'jobId': job_id},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logs = self.get_logs(job_id)
                
                error = None
                if result.get('state') in ['Error', 'Failed']:
                    error = result.get('jobError', {}).get('message', 'Unknown error')
                
                duration = None
                if result.get('startedDateTime') and result.get('finishedDateTime'):
                    from datetime import datetime
                    start = datetime.fromisoformat(result['startedDateTime'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(result['finishedDateTime'].replace('Z', '+00:00'))
                    duration = (end - start).total_seconds()
                
                return JobResult(
                    job_id=job_id,
                    status=self.get_job_status(job_id),
                    output=logs,
                    error=error,
                    duration_seconds=duration
                )
            else:
                raise CloudException(f"Failed to get job result: {response.text}")
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running Paperspace job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            response = requests.post(
                f"{self.base_url}/jobs/stop",
                headers=headers,
                json={'jobId': job_id},
                timeout=30
            )
            
            return response.status_code in [200, 201, 204]
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """List recent Paperspace jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'X-API-Key': self.api_key}
            params = {}
            if self.project_id:
                params['projectId'] = self.project_id
            
            response = requests.get(
                f"{self.base_url}/jobs/getJobs",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_list = result if isinstance(result, list) else result.get('jobs', [])
                
                jobs = []
                for job in job_list[:limit]:
                    job_status = self._map_status(job.get('state', 'Unknown'))
                    
                    if status_filter is None or job_status == status_filter:
                        jobs.append({
                            'job_id': job.get('id') or job.get('jobId'),
                            'job_name': job.get('name', ''),
                            'status': job_status,
                            'start_time': job.get('startedDateTime'),
                            'end_time': job.get('finishedDateTime'),
                        })
                
                return jobs
            else:
                raise CloudException(f"Failed to list jobs: {response.text}")
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a Paperspace job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {'X-API-Key': self.api_key}
            response = requests.get(
                f"{self.base_url}/jobs/logs",
                headers=headers,
                params={'jobId': job_id},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.text
            else:
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
        """Deploy a model to Paperspace Gradient deployment."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            resource_config = resource_config or ResourceConfig(
                instance_type='P4000',
                gpu_enabled=True
            )
            
            deployment_config = {
                "deploymentType": "TFServing",
                "imageUrl": "tensorflow/serving:latest-gpu" if resource_config.gpu_enabled else "tensorflow/serving:latest",
                "name": endpoint_name,
                "machineType": resource_config.instance_type,
                "instanceCount": 1,
                "modelId": model_path,
            }
            
            if self.project_id:
                deployment_config["projectId"] = self.project_id
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/deployments/createDeployment",
                headers=headers,
                json=deployment_config,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                deployment_id = result.get('id')
                endpoint_url = result.get('endpoint', f"https://{deployment_id}.gradient.paperspace.com")
                logger.info(f"Model deployed to: {endpoint_url}")
                return endpoint_url
            else:
                raise CloudException(f"Model deployment failed: {response.text}")
        except Exception as e:
            raise CloudException(f"Deployment failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a Paperspace deployment."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import requests
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            response = requests.post(
                f"{self.base_url}/deployments/deleteDeployment",
                headers=headers,
                json={'id': endpoint_name},
                timeout=30
            )
            
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def _build_command(self, code: str, dependencies: Optional[List[str]]) -> str:
        """Build command to execute code."""
        commands = [
            "pip install neural-dsl",
        ]
        
        if dependencies:
            commands.append(f"pip install {' '.join(dependencies)}")
        
        commands.append(f"python -c '{code}'")
        
        return " && ".join(commands)
    
    def _map_status(self, state: str) -> JobStatus:
        """Map Paperspace state to JobStatus."""
        status_map = {
            'Pending': JobStatus.PENDING,
            'Running': JobStatus.RUNNING,
            'Stopped': JobStatus.SUCCEEDED,
            'Error': JobStatus.FAILED,
            'Failed': JobStatus.FAILED,
            'Cancelled': JobStatus.CANCELLED,
        }
        return status_map.get(state, JobStatus.UNKNOWN)
