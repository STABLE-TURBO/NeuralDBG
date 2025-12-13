"""
AWS SageMaker platform connector for Neural DSL.

Provides integration with SageMaker Studio, training jobs, and model deployment.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from neural.exceptions import CloudConnectionError, CloudException, CloudExecutionError

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


logger = logging.getLogger(__name__)


class SageMakerConnector(BaseConnector):
    """
    Connector for AWS SageMaker platform.
    
    Features:
    - Submit training jobs
    - Deploy models to SageMaker endpoints
    - Execute code in SageMaker Studio notebooks
    - Manage compute resources
    
    Authentication:
        Uses AWS credentials (access_key_id, secret_access_key, region)
        or IAM role when running on AWS infrastructure.
    """
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker connector.
        
        Args:
            credentials: Dictionary with AWS credentials or None for IAM role
        """
        super().__init__(credentials)
        self.region = self.credentials.get('region', 'us-east-1')
        self.role_arn = self.credentials.get('role_arn')
        self._sagemaker_client = None
        self._s3_client = None
        
    def authenticate(self) -> bool:
        """Authenticate with AWS SageMaker."""
        try:
            import boto3
            
            session_kwargs = {}
            if 'access_key_id' in self.credentials:
                session_kwargs['aws_access_key_id'] = self.credentials['access_key_id']
            if 'secret_access_key' in self.credentials:
                session_kwargs['aws_secret_access_key'] = self.credentials['secret_access_key']
            if self.region:
                session_kwargs['region_name'] = self.region
            
            session = boto3.Session(**session_kwargs)
            self._sagemaker_client = session.client('sagemaker')
            self._s3_client = session.client('s3')
            
            self._sagemaker_client.list_training_jobs(MaxResults=1)
            
            if not self.role_arn:
                iam = session.client('iam')
                roles = iam.list_roles()
                for role in roles['Roles']:
                    if 'SageMaker' in role['RoleName']:
                        self.role_arn = role['Arn']
                        break
                
                if not self.role_arn:
                    logger.warning("No SageMaker execution role found")
            
            self.authenticated = True
            logger.info("Successfully authenticated with AWS SageMaker")
            return True
            
        except ImportError:
            raise CloudConnectionError(
                "boto3 library required. Install with: pip install boto3"
            )
        except Exception as e:
            raise CloudConnectionError(f"SageMaker authentication failed: {e}")
    
    def submit_job(
        self,
        code: str,
        resource_config: Optional[ResourceConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        job_name: Optional[str] = None,
    ) -> str:
        """Submit a training job to SageMaker."""
        if not self.authenticated:
            self.authenticate()
        
        if not self.role_arn:
            raise CloudExecutionError("SageMaker execution role not configured")
        
        try:
            job_name = job_name or f"neural-job-{int(time.time())}"
            
            s3_bucket = self.credentials.get('s3_bucket', f'sagemaker-{self.region}')
            s3_key = f"neural-jobs/{job_name}/source.tar.gz"
            
            source_dir = self._prepare_training_code(code, dependencies)
            self._upload_to_s3(source_dir, s3_bucket, s3_key)
            
            resource_config = resource_config or ResourceConfig(
                instance_type='ml.m5.large',
                gpu_enabled=False
            )
            
            training_config = {
                'TrainingJobName': job_name,
                'RoleArn': self.role_arn,
                'AlgorithmSpecification': {
                    'TrainingImage': self._get_training_image(resource_config.gpu_enabled),
                    'TrainingInputMode': 'File',
                },
                'ResourceConfig': {
                    'InstanceType': resource_config.instance_type,
                    'InstanceCount': 1,
                    'VolumeSizeInGB': resource_config.disk_size_gb or 30,
                },
                'StoppingCondition': {
                    'MaxRuntimeInSeconds': (resource_config.max_runtime_hours or 24) * 3600,
                },
                'OutputDataConfig': {
                    'S3OutputPath': f's3://{s3_bucket}/neural-jobs/{job_name}/output',
                },
            }
            
            if environment:
                training_config['Environment'] = environment
            
            response = self._sagemaker_client.create_training_job(**training_config)
            
            logger.info(f"Training job submitted: {job_name}")
            return job_name
            
        except Exception as e:
            raise CloudExecutionError(f"Job submission failed: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a SageMaker training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_id
            )
            
            status = response.get('TrainingJobStatus', 'Unknown')
            
            status_map = {
                'InProgress': JobStatus.RUNNING,
                'Completed': JobStatus.SUCCEEDED,
                'Failed': JobStatus.FAILED,
                'Stopping': JobStatus.RUNNING,
                'Stopped': JobStatus.CANCELLED,
            }
            
            return status_map.get(status, JobStatus.UNKNOWN)
            
        except Exception as e:
            raise CloudException(f"Status retrieval failed: {e}")
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a SageMaker training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_id
            )
            
            status = self.get_job_status(job_id)
            
            output_path = None
            if 'ModelArtifacts' in response:
                output_path = response['ModelArtifacts'].get('S3ModelArtifacts')
            
            error = None
            if status == JobStatus.FAILED:
                error = response.get('FailureReason', 'Unknown error')
            
            metrics = {}
            if 'FinalMetricDataList' in response:
                for metric in response['FinalMetricDataList']:
                    metrics[metric['MetricName']] = metric['Value']
            
            duration = None
            if 'TrainingStartTime' in response and 'TrainingEndTime' in response:
                start = response['TrainingStartTime'].timestamp()
                end = response['TrainingEndTime'].timestamp()
                duration = end - start
            
            return JobResult(
                job_id=job_id,
                status=status,
                output=output_path,
                error=error,
                metrics=metrics,
                artifacts=[output_path] if output_path else [],
                duration_seconds=duration
            )
            
        except Exception as e:
            raise CloudException(f"Result retrieval failed: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running SageMaker training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            self._sagemaker_client.stop_training_job(TrainingJobName=job_id)
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
        """List recent SageMaker training jobs."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            params = {'MaxResults': limit, 'SortBy': 'CreationTime', 'SortOrder': 'Descending'}
            
            if status_filter:
                status_map_reverse = {
                    JobStatus.RUNNING: 'InProgress',
                    JobStatus.SUCCEEDED: 'Completed',
                    JobStatus.FAILED: 'Failed',
                    JobStatus.CANCELLED: 'Stopped',
                }
                if status_filter in status_map_reverse:
                    params['StatusEquals'] = status_map_reverse[status_filter]
            
            response = self._sagemaker_client.list_training_jobs(**params)
            
            jobs = []
            for job in response.get('TrainingJobSummaries', []):
                jobs.append({
                    'job_id': job['TrainingJobName'],
                    'job_name': job['TrainingJobName'],
                    'status': self._map_status(job['TrainingJobStatus']),
                    'start_time': job.get('CreationTime'),
                    'end_time': job.get('TrainingEndTime'),
                })
            
            return jobs
            
        except Exception as e:
            raise CloudException(f"Job listing failed: {e}")
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a SageMaker training job."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            import boto3
            
            logs_client = boto3.client('logs', region_name=self.region)
            log_group = '/aws/sagemaker/TrainingJobs'
            log_stream = job_id
            
            response = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startFromHead=True
            )
            
            logs = []
            for event in response.get('events', []):
                logs.append(event['message'])
            
            return '\n'.join(logs)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve logs: {e}")
            return "Logs not available"
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        resource_config: Optional[ResourceConfig] = None
    ) -> str:
        """Deploy a model to a SageMaker endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        if not self.role_arn:
            raise CloudExecutionError("SageMaker execution role not configured")
        
        try:
            resource_config = resource_config or ResourceConfig(
                instance_type='ml.t2.medium',
                gpu_enabled=False
            )
            
            model_name = f"{endpoint_name}-model"
            
            self._sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': self._get_inference_image(resource_config.gpu_enabled),
                    'ModelDataUrl': model_path,
                },
                ExecutionRoleArn=self.role_arn
            )
            
            endpoint_config_name = f"{endpoint_name}-config"
            self._sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': resource_config.instance_type,
                }]
            )
            
            self._sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"Model deployment initiated: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            raise CloudException(f"Model deployment failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a SageMaker endpoint."""
        if not self.authenticated:
            self.authenticate()
        
        try:
            self._sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def _prepare_training_code(
        self,
        code: str,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Prepare training code package."""
        import tarfile
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        
        train_script = temp_dir / "train.py"
        train_script.write_text(f"""
import os
import sys

# Install Neural DSL
os.system('pip install neural-dsl')

{f"# Install dependencies\\nos.system('pip install {' '.join(dependencies)}')" if dependencies else ""}

# Execute Neural DSL code
{code}
""")
        
        tar_path = temp_dir / "source.tar.gz"
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(train_script, arcname='train.py')
        
        return str(tar_path)
    
    def _upload_to_s3(self, local_path: str, bucket: str, key: str) -> None:
        """Upload file to S3."""
        self._s3_client.upload_file(local_path, bucket, key)
    
    def _get_training_image(self, gpu_enabled: bool) -> str:
        """Get SageMaker training container image."""
        if gpu_enabled:
            return f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-training:2.12-gpu-py310"
        else:
            return f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-training:2.12-cpu-py310"
    
    def _get_inference_image(self, gpu_enabled: bool) -> str:
        """Get SageMaker inference container image."""
        if gpu_enabled:
            return f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-inference:2.12-gpu"
        else:
            return f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-inference:2.12-cpu"
    
    def _map_status(self, status: str) -> JobStatus:
        """Map SageMaker status to JobStatus."""
        status_map = {
            'InProgress': JobStatus.RUNNING,
            'Completed': JobStatus.SUCCEEDED,
            'Failed': JobStatus.FAILED,
            'Stopping': JobStatus.RUNNING,
            'Stopped': JobStatus.CANCELLED,
        }
        return status_map.get(status, JobStatus.UNKNOWN)
