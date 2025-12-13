"""
Run:AI platform connector for Neural DSL.

Provides integration with Run:AI Kubernetes orchestration for GPU workloads.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class RunAIConnector(BaseConnector):
    """
    Connector for Run:AI platform.
    
    Features:
    - Submit jobs to Run:AI clusters
    - Manage GPU resources and quotas
    - Execute distributed training jobs
    - Monitor job status and logs
    
    Authentication:
        Requires Run:AI CLI to be installed and configured.
        Credentials include 'cluster_url' and 'kubeconfig' path.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize Run:AI connector.
        
        Args:
            credentials: Dictionary with cluster credentials
        """
        super().__init__(credentials)
        self.cluster_url = self.credentials.get('cluster_url')
        self.kubeconfig = self.credentials.get('kubeconfig')
        self.project = self.credentials.get('project', 'default')
        self._cli_available = False
        
    def authenticate(self) -> bool:
        """Authenticate with Run:AI cluster."""
        try:
            result = subprocess.run(
                ['runai', 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise CloudConnectionError(
                    "Run:AI CLI not found. Install from: https://docs.run.ai/latest/admin/runai-setup/cli-install/"
                )
            
            self._cli_available = True
            
            if self.project:
                result = subprocess.run(
                    ['runai', 'config', 'project', self.project],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            result = subprocess.run(
                ['runai', 'list', 'jobs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.authenticated = True
                logger.info(f"Successfully authenticated with Run:AI (project: {self.project})")
                return True
            else:
                raise CloudConnectionError(
                    f"Run:AI authentication failed: {result.stderr}"
                )
                
        except FileNotFoundError:
            raise CloudConnectionError(
                "Run:AI CLI not found. Install from: https://docs.run.ai/latest/admin/runai-setup/cli-install/"
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
        """Submit a job to Run:AI."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            job_name = job_name or f"neural-job-{int(time.time())}"
            
            resource_config = resource_config or ResourceConfig(
                instance_type='V100',
                gpu_enabled=True,
                gpu_count=1
            )
            
            script_path = self._prepare_training_script(code, dependencies)
            
            command = [
                'runai', 'submit', job_name,
                '--image', 'tensorflow/tensorflow:latest-gpu',
                '--gpu', str(resource_config.gpu_count),
                '--volume', f'{script_path}:/workspace',
                '--command', '--',
                'python', '/workspace/train.py'
            ]
            
            if environment:
                for key, value in environment.items():
                    command.extend(['-e', f'{key}={value}'])
            
            if resource_config.memory_gb:
                command.extend(['--memory', f'{resource_config.memory_gb}G'])
            
            if resource_config.cpu_count:
                command.extend(['--cpu', str(resource_config.cpu_count)])
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Job submitted successfully: {job_name}")
                return job_name
            else:
                raise CloudExecutionError(
                    f"Failed to submit job: {result.stderr}"
                )
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a Run:AI job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            result = subprocess.run(
                ['runai', 'describe', 'job', job_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                if 'Running' in output:
                    return JobStatus.RUNNING
                elif 'Succeeded' in output or 'Completed' in output:
                    return JobStatus.SUCCEEDED
                elif 'Failed' in output or 'Error' in output:
                    return JobStatus.FAILED
                elif 'Pending' in output or 'Creating' in output:
                    return JobStatus.PENDING
                else:
                    return JobStatus.UNKNOWN
            else:
                raise CloudException(f"Failed to get job status: {result.stderr}")
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a Run:AI job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            status = self.get_job_status(job_id)
            
            result = subprocess.run(
                ['runai', 'describe', 'job', job_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            error = None
            if status == JobStatus.FAILED:
                if 'Error' in result.stdout or 'Failed' in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Error' in line or 'Failed' in line:
                            error = line.strip()
                            break
            
            logs = self.get_logs(job_id)
            
            return JobResult(
                job_id=job_id,
                status=status,
                output=logs,
                error=error
            )
            
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running Run:AI job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            result = subprocess.run(
                ['runai', 'delete', 'job', job_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Job cancelled: {job_id}")
                return True
            else:
                logger.error(f"Failed to cancel job: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(
        self,
        limit: int = 10,
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """List recent Run:AI jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            result = subprocess.run(
                ['runai', 'list', 'jobs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) < 2:
                    return []
                
                jobs = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        job_name = parts[0]
                        status_str = parts[1] if len(parts) > 1 else 'Unknown'
                        
                        job_status = self._map_status(status_str)
                        
                        if status_filter is None or job_status == status_filter:
                            jobs.append({
                                'job_id': job_name,
                                'job_name': job_name,
                                'status': job_status,
                                'start_time': None,
                                'end_time': None,
                            })
                            
                            if len(jobs) >= limit:
                                break
                
                return jobs
            else:
                raise CloudException(f"Failed to list jobs: {result.stderr}")
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a Run:AI job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            result = subprocess.run(
                ['runai', 'logs', job_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return "Logs not available"
        except Exception as e:
            logger.warning(f"Failed to retrieve logs: {e}")
            return "Logs not available"
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current GPU resource usage."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            result = subprocess.run(
                ['runai', 'list', 'jobs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_gpus = 0
                used_gpus = 0
                
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            gpu_count = int(parts[3]) if parts[3].isdigit() else 0
                            total_gpus += gpu_count
                            if 'Running' in line:
                                used_gpus += gpu_count
                        except:
                            pass
                
                return {
                    'total_gpus': total_gpus,
                    'used_gpus': used_gpus,
                    'available_gpus': total_gpus - used_gpus,
                    'project': self.project
                }
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}
    
    def _prepare_training_script(
        self,
        code: str,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Prepare training script."""
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        
        train_script = temp_dir / "train.py"
        
        script_content = "#!/usr/bin/env python\n"
        script_content += "import os\nimport sys\n\n"
        
        if dependencies:
            script_content += f"os.system('pip install {' '.join(dependencies)}')\n"
        
        script_content += "os.system('pip install neural-dsl')\n\n"
        script_content += f"{code}\n"
        
        train_script.write_text(script_content)
        
        return str(temp_dir)
    
    def _map_status(self, status: str) -> JobStatus:
        """Map Run:AI status to JobStatus."""
        status = status.lower()
        
        if 'running' in status:
            return JobStatus.RUNNING
        elif 'succeeded' in status or 'completed' in status:
            return JobStatus.SUCCEEDED
        elif 'failed' in status or 'error' in status:
            return JobStatus.FAILED
        elif 'pending' in status or 'creating' in status:
            return JobStatus.PENDING
        else:
            return JobStatus.UNKNOWN
