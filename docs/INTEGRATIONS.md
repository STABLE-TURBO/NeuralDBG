# Neural DSL Platform Integrations

Complete guide to using Neural DSL with popular ML platforms.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Platforms](#supported-platforms)
- [Usage Patterns](#usage-patterns)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Neural DSL integrations module provides a unified interface to execute Neural DSL models on popular ML platforms. It handles authentication, job submission, resource management, and model deployment across different cloud providers.

### Key Features

- **Unified API**: Same interface across all platforms
- **Authentication**: Secure credential management
- **Remote Execution**: Submit training jobs to cloud resources
- **Resource Management**: Configure compute, GPU, and memory
- **Model Deployment**: Deploy models as endpoints
- **Job Monitoring**: Track status and retrieve logs
- **Cost Estimation**: Estimate resource costs

## Installation

### Basic Installation

```bash
pip install neural-dsl[integrations]
```

### Platform-Specific Installation

```bash
# Databricks
pip install requests

# AWS SageMaker
pip install boto3

# Google Vertex AI
pip install google-cloud-aiplatform google-cloud-storage

# Azure ML
pip install azure-ai-ml azure-identity

# Paperspace
pip install requests

# Run:AI (requires CLI)
# See: https://docs.run.ai/latest/admin/runai-setup/cli-install/
```

### Full Installation

```bash
pip install neural-dsl[full]
```

## Quick Start

```python
from neural.integrations import PlatformManager, ResourceConfig

# Initialize manager
manager = PlatformManager()

# Configure platform
manager.configure_platform(
    'databricks',
    host='https://your-workspace.cloud.databricks.com',
    token='dapi...'
)

# Define Neural DSL model
dsl_code = """
model MyModel {
    input: (None, 28, 28, 1)
    Conv2D(filters=32, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=10, activation='softmax')
}
"""

# Submit job
job_id = manager.submit_job(
    platform='databricks',
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='i3.xlarge',
        gpu_enabled=False
    ),
    job_name='my-training-job'
)

# Check status
status = manager.get_job_status('databricks', job_id)
print(f"Job status: {status}")

# Get results
result = manager.get_job_result('databricks', job_id)
print(f"Output: {result.output}")
```

## Supported Platforms

### Databricks

**Authentication:**
- Host URL
- Personal access token
- Optional cluster ID

**Features:**
- Notebook execution
- Cluster management
- Model serving
- Job scheduling

**Example:**
```python
manager.configure_platform(
    'databricks',
    host='https://your-workspace.cloud.databricks.com',
    token='dapi...',
    cluster_id='1234-567890-abc123'  # Optional
)
```

### AWS SageMaker

**Authentication:**
- AWS access key ID
- AWS secret access key
- Region
- IAM role ARN
- S3 bucket

**Features:**
- Training jobs
- Endpoint deployment
- Model registry
- Batch transform

**Example:**
```python
manager.configure_platform(
    'sagemaker',
    access_key_id='AKIA...',
    secret_access_key='wJalr...',
    region='us-east-1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    s3_bucket='my-bucket'
)
```

### Google Vertex AI

**Authentication:**
- Project ID
- Location/Region
- Service account credentials file

**Features:**
- Custom training jobs
- Endpoint deployment
- Workbench notebooks
- AutoML integration

**Example:**
```python
manager.configure_platform(
    'vertex_ai',
    project_id='my-gcp-project',
    location='us-central1',
    credentials_file='/path/to/service-account.json'
)
```

### Azure ML Studio

**Authentication:**
- Subscription ID
- Resource group
- Workspace name

**Features:**
- Compute clusters
- Pipeline execution
- Model deployment
- Experiment tracking

**Example:**
```python
manager.configure_platform(
    'azure_ml',
    subscription_id='12345678-1234-1234-1234-123456789012',
    resource_group='my-resource-group',
    workspace_name='my-workspace'
)
```

### Paperspace Gradient

**Authentication:**
- API key
- Optional project ID

**Features:**
- Job execution
- Notebook environment
- Model deployment
- GPU access

**Example:**
```python
manager.configure_platform(
    'paperspace',
    api_key='ps_...',
    project_id='prj...'
)
```

### Run:AI

**Authentication:**
- Kubernetes cluster
- Run:AI CLI configuration
- Project name

**Features:**
- GPU orchestration
- Distributed training
- Resource quotas
- Job scheduling

**Example:**
```python
manager.configure_platform(
    'runai',
    cluster_url='https://my-cluster.run.ai',
    project='my-project'
)
```

## Usage Patterns

### Single Platform

```python
from neural.integrations import DatabricksConnector, ResourceConfig

connector = DatabricksConnector(credentials={
    'host': '...',
    'token': '...'
})

connector.authenticate()

job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='i3.xlarge',
        gpu_enabled=False
    )
)

status = connector.get_job_status(job_id)
```

### Multiple Platforms

```python
from neural.integrations import PlatformManager

manager = PlatformManager()

# Configure multiple platforms
manager.configure_platform('databricks', host='...', token='...')
manager.configure_platform('sagemaker', access_key_id='...', secret_access_key='...')

# Submit to different platforms
db_job = manager.submit_job('databricks', code=code1)
sm_job = manager.submit_job('sagemaker', code=code2)

# Monitor all jobs
print(f"Databricks: {manager.get_job_status('databricks', db_job)}")
print(f"SageMaker: {manager.get_job_status('sagemaker', sm_job)}")
```

### Batch Processing

```python
from neural.integrations.utils import batch_submit_jobs

jobs = [
    {'code': code1, 'job_name': 'job-1'},
    {'code': code2, 'job_name': 'job-2'},
    {'code': code3, 'job_name': 'job-3'},
]

job_ids = batch_submit_jobs(manager, 'databricks', jobs)
```

### Waiting for Completion

```python
from neural.integrations.utils import wait_for_job_completion

result = wait_for_job_completion(
    manager,
    platform='databricks',
    job_id=job_id,
    poll_interval=30,
    timeout=3600
)

print(f"Job completed: {result.status}")
```

### Cost Estimation

```python
from neural.integrations.utils import estimate_resource_cost

cost = estimate_resource_cost(
    platform='sagemaker',
    instance_type='ml.p3.2xlarge',
    duration_hours=2.0,
    gpu_enabled=True
)

print(f"Estimated cost: ${cost:.2f}")
```

## API Reference

### PlatformManager

```python
class PlatformManager:
    def configure_platform(platform, credentials=None, **kwargs) -> bool
    def set_active_platform(platform) -> bool
    def submit_job(platform=None, code, resource_config=None, ...) -> str
    def get_job_status(platform=None, job_id) -> JobStatus
    def get_job_result(platform=None, job_id) -> JobResult
    def cancel_job(platform=None, job_id) -> bool
    def list_jobs(platform=None, limit=10, status_filter=None) -> List[Dict]
    def get_logs(platform=None, job_id) -> str
    def deploy_model(platform=None, model_path, endpoint_name, ...) -> str
    def delete_endpoint(platform=None, endpoint_name) -> bool
    def list_platforms() -> List[str]
    def list_configured_platforms() -> List[str]
    def get_platform_info(platform) -> Dict
```

### BaseConnector

```python
class BaseConnector(ABC):
    def authenticate() -> bool
    def submit_job(code, resource_config=None, ...) -> str
    def get_job_status(job_id) -> JobStatus
    def get_job_result(job_id) -> JobResult
    def cancel_job(job_id) -> bool
    def list_jobs(limit=10, status_filter=None) -> List[Dict]
    def get_logs(job_id) -> str
    def deploy_model(model_path, endpoint_name, ...) -> str
    def delete_endpoint(endpoint_name) -> bool
    def upload_file(local_path, remote_path) -> bool
    def download_file(remote_path, local_path) -> bool
    def get_resource_usage() -> Dict
```

### ResourceConfig

```python
@dataclass
class ResourceConfig:
    instance_type: str
    gpu_enabled: bool = False
    gpu_count: int = 0
    memory_gb: Optional[int] = None
    cpu_count: Optional[int] = None
    disk_size_gb: Optional[int] = None
    max_runtime_hours: Optional[int] = None
    auto_shutdown: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### JobStatus

```python
class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
```

### JobResult

```python
@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    logs_url: Optional[str] = None
    duration_seconds: Optional[float] = None
```

## Examples

See `neural/integrations/examples.py` for complete examples including:

- Unified manager usage
- Platform-specific configurations
- Model training workflows
- Deployment patterns
- Resource management
- Batch processing

## Troubleshooting

### Authentication Issues

**Problem**: Authentication fails with credential errors.

**Solutions**:
- Verify credentials are correct
- Check environment variables
- Ensure service accounts have proper permissions
- Test credentials with platform's native CLI

### Job Submission Failures

**Problem**: Jobs fail to submit or start.

**Solutions**:
- Check resource availability
- Verify instance types are valid for platform
- Ensure code has no syntax errors
- Check platform-specific quotas and limits

### Resource Unavailable

**Problem**: Requested resources not available.

**Solutions**:
- Try different instance types
- Check regional availability
- Use smaller resource configurations
- Enable auto-scaling if supported

### Connection Timeouts

**Problem**: Operations timeout or hang.

**Solutions**:
- Check network connectivity
- Verify firewall rules
- Increase timeout values
- Use async operations for long-running tasks

### Cost Management

**Problem**: Unexpected high costs.

**Solutions**:
- Enable auto-shutdown for resources
- Use spot/preemptible instances when possible
- Set max_runtime_hours limits
- Monitor resource usage regularly
- Use cost estimation before submission

## Best Practices

1. **Credential Security**
   - Never hardcode credentials
   - Use environment variables or config files
   - Rotate keys regularly
   - Use IAM roles when possible

2. **Resource Management**
   - Always set auto_shutdown=True
   - Use appropriate instance types
   - Set max_runtime_hours to prevent runaway jobs
   - Monitor resource usage

3. **Error Handling**
   - Always wrap operations in try-except blocks
   - Check job status before retrieving results
   - Log errors for debugging
   - Implement retry logic for transient failures

4. **Performance**
   - Use batch operations when possible
   - Configure appropriate resource sizes
   - Enable GPU only when needed
   - Monitor and optimize job duration

5. **Cost Optimization**
   - Estimate costs before submission
   - Use cheaper instance types for development
   - Leverage free tiers and credits
   - Clean up unused resources

## Additional Resources

- [Neural DSL Documentation](../README.md)
- [Platform-specific READMEs](./README.md)
- [Example Scripts](../neural/integrations/examples.py)
- [API Reference](../neural/integrations/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: https://github.com/Lemniscate-world/Neural
