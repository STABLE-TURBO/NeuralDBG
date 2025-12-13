# Neural DSL Platform Integrations

Comprehensive integration connectors for popular ML platforms with unified API for authentication, remote execution, and resource management.

## Supported Platforms

- **Databricks**: Notebooks, clusters, and jobs API
- **AWS SageMaker**: Studio, training jobs, and model deployment
- **Google Vertex AI**: Training, prediction, and workbench notebooks
- **Azure ML Studio**: Workspaces, compute clusters, and endpoints
- **Paperspace Gradient**: Notebooks, jobs, and deployments
- **Run:AI**: Kubernetes orchestration for GPU workloads

## Quick Start

### Using the Unified Manager

```python
from neural.integrations import PlatformManager, ResourceConfig

# Initialize manager
manager = PlatformManager()

# Configure a platform
manager.configure_platform(
    'databricks',
    host='https://your-workspace.cloud.databricks.com',
    token='your-token'
)

# Submit a job
job_id = manager.submit_job(
    platform='databricks',
    code="""
model MyModel {
    input: (None, 28, 28, 1)
    Conv2D(filters=32, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=10, activation='softmax')
}
""",
    resource_config=ResourceConfig(
        instance_type='i3.xlarge',
        gpu_enabled=False
    )
)

# Check status
status = manager.get_job_status('databricks', job_id)
print(f"Job status: {status}")

# Get results
result = manager.get_job_result('databricks', job_id)
print(f"Output: {result.output}")
```

### Using Individual Connectors

```python
from neural.integrations import DatabricksConnector, ResourceConfig

# Initialize connector
connector = DatabricksConnector(credentials={
    'host': 'https://your-workspace.cloud.databricks.com',
    'token': 'your-token',
    'cluster_id': 'your-cluster-id'  # Optional
})

# Authenticate
connector.authenticate()

# Submit job
job_id = connector.submit_job(
    code="print('Hello from Databricks!')",
    resource_config=ResourceConfig(
        instance_type='i3.xlarge',
        gpu_enabled=False
    )
)
```

## Platform-Specific Guides

### Databricks

```python
from neural.integrations import DatabricksConnector

connector = DatabricksConnector(credentials={
    'host': 'https://your-workspace.cloud.databricks.com',
    'token': 'dapi...',
    'cluster_id': '1234-567890-abc123'  # Optional
})

connector.authenticate()

# Deploy model to Model Serving
endpoint_url = connector.deploy_model(
    model_path='dbfs:/models/my-model',
    endpoint_name='my-neural-model'
)
```

### AWS SageMaker

```python
from neural.integrations import SageMakerConnector

connector = SageMakerConnector(credentials={
    'access_key_id': 'AKIA...',
    'secret_access_key': 'wJalr...',
    'region': 'us-east-1',
    'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole',
    's3_bucket': 'my-sagemaker-bucket'
})

connector.authenticate()

# Submit training job
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='ml.p3.2xlarge',
        gpu_enabled=True
    )
)
```

### Google Vertex AI

```python
from neural.integrations import VertexAIConnector

connector = VertexAIConnector(credentials={
    'project_id': 'my-gcp-project',
    'location': 'us-central1',
    'credentials_file': '/path/to/service-account.json'  # Optional
})

connector.authenticate()

# Submit custom training job
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='n1-standard-4',
        gpu_enabled=True,
        gpu_count=1
    )
)
```

### Azure ML Studio

```python
from neural.integrations import AzureMLConnector

connector = AzureMLConnector(credentials={
    'subscription_id': '12345678-1234-1234-1234-123456789012',
    'resource_group': 'my-resource-group',
    'workspace_name': 'my-workspace'
})

connector.authenticate()

# Submit training job
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='Standard_NC6',
        gpu_enabled=True
    )
)
```

### Paperspace Gradient

```python
from neural.integrations import PaperspaceConnector

connector = PaperspaceConnector(credentials={
    'api_key': 'ps_...',
    'project_id': 'prj...'  # Optional
})

connector.authenticate()

# Submit job
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='P4000',
        gpu_enabled=True
    )
)
```

### Run:AI

```python
from neural.integrations import RunAIConnector

connector = RunAIConnector(credentials={
    'cluster_url': 'https://my-cluster.run.ai',
    'kubeconfig': '/path/to/kubeconfig',  # Optional
    'project': 'my-project'
})

connector.authenticate()

# Submit distributed training job
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=ResourceConfig(
        instance_type='V100',
        gpu_enabled=True,
        gpu_count=4
    )
)
```

## Resource Configuration

```python
from neural.integrations import ResourceConfig

# CPU-only configuration
cpu_config = ResourceConfig(
    instance_type='n1-standard-4',
    gpu_enabled=False,
    memory_gb=16,
    cpu_count=4,
    disk_size_gb=100
)

# GPU configuration
gpu_config = ResourceConfig(
    instance_type='p3.2xlarge',
    gpu_enabled=True,
    gpu_count=1,
    memory_gb=64,
    max_runtime_hours=24,
    auto_shutdown=True
)
```

## Job Management

### Submitting Jobs

```python
job_id = manager.submit_job(
    platform='sagemaker',
    code=dsl_code,
    resource_config=gpu_config,
    environment={'PYTHONPATH': '/app'},
    dependencies=['tensorflow>=2.12', 'numpy>=1.23'],
    job_name='my-training-job'
)
```

### Monitoring Jobs

```python
# Check status
status = manager.get_job_status('sagemaker', job_id)

# Get full result
result = manager.get_job_result('sagemaker', job_id)
print(f"Status: {result.status}")
print(f"Output: {result.output}")
print(f"Metrics: {result.metrics}")
print(f"Duration: {result.duration_seconds}s")

# Get logs
logs = manager.get_logs('sagemaker', job_id)
print(logs)
```

### Listing Jobs

```python
from neural.integrations import JobStatus

# List all jobs
jobs = manager.list_jobs('databricks', limit=20)

# Filter by status
running_jobs = manager.list_jobs(
    'databricks',
    limit=10,
    status_filter=JobStatus.RUNNING
)
```

### Canceling Jobs

```python
success = manager.cancel_job('databricks', job_id)
if success:
    print("Job cancelled successfully")
```

## Model Deployment

```python
# Deploy model
endpoint_url = manager.deploy_model(
    platform='vertex_ai',
    model_path='gs://my-bucket/models/my-model',
    endpoint_name='neural-model-prod',
    resource_config=ResourceConfig(
        instance_type='n1-standard-2',
        gpu_enabled=False
    )
)

# Delete endpoint
manager.delete_endpoint('vertex_ai', 'neural-model-prod')
```

## Resource Management

```python
# Get resource usage
usage = manager.get_resource_usage('runai')
print(f"Total GPUs: {usage['total_gpus']}")
print(f"Used GPUs: {usage['used_gpus']}")
print(f"Available GPUs: {usage['available_gpus']}")
```

## Error Handling

```python
from neural.exceptions import (
    CloudConnectionError,
    CloudExecutionError,
    CloudException
)

try:
    manager.configure_platform('databricks', host='...', token='...')
except CloudConnectionError as e:
    print(f"Authentication failed: {e}")

try:
    job_id = manager.submit_job('databricks', code=dsl_code)
except CloudExecutionError as e:
    print(f"Job submission failed: {e}")
```

## Platform Information

```python
# List all available platforms
platforms = manager.list_platforms()
print(f"Available platforms: {platforms}")

# List configured platforms
configured = manager.list_configured_platforms()
print(f"Configured platforms: {configured}")

# Get platform info
info = manager.get_platform_info('databricks')
print(f"Name: {info['name']}")
print(f"Configured: {info['configured']}")
print(f"Active: {info['active']}")

# Get all platform info
all_info = manager.get_all_platform_info()
for platform in all_info:
    print(f"{platform['name']}: {platform['description']}")
```

## Advanced Usage

### Multiple Platforms

```python
# Configure multiple platforms
manager.configure_platform('databricks', host='...', token='...')
manager.configure_platform('sagemaker', access_key_id='...', secret_access_key='...')

# Switch between platforms
manager.set_active_platform('databricks')
job1 = manager.submit_job(code=code1)  # Runs on Databricks

manager.set_active_platform('sagemaker')
job2 = manager.submit_job(code=code2)  # Runs on SageMaker
```

### Custom Resource Configurations

```python
# Different configs for different platforms
databricks_config = ResourceConfig(
    instance_type='i3.xlarge',
    gpu_enabled=False
)

sagemaker_config = ResourceConfig(
    instance_type='ml.p3.2xlarge',
    gpu_enabled=True,
    gpu_count=1
)

# Submit to different platforms
db_job = manager.submit_job('databricks', code=code, resource_config=databricks_config)
sm_job = manager.submit_job('sagemaker', code=code, resource_config=sagemaker_config)
```

## Dependencies

Install platform-specific dependencies:

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

# Run:AI
# Install Run:AI CLI from https://docs.run.ai/latest/admin/runai-setup/cli-install/
```

Or install all at once:

```bash
pip install neural-dsl[cloud]
```

## API Reference

### BaseConnector

Abstract base class for all platform connectors.

**Methods:**
- `authenticate()`: Authenticate with the platform
- `submit_job()`: Submit a job for execution
- `get_job_status()`: Get job status
- `get_job_result()`: Get job result
- `cancel_job()`: Cancel a job
- `list_jobs()`: List recent jobs
- `get_logs()`: Get job logs
- `deploy_model()`: Deploy a model
- `delete_endpoint()`: Delete an endpoint
- `upload_file()`: Upload a file
- `download_file()`: Download a file
- `get_resource_usage()`: Get resource usage

### PlatformManager

Unified manager for all platforms.

**Methods:**
- `configure_platform()`: Configure a platform
- `set_active_platform()`: Set active platform
- `submit_job()`: Submit a job
- `get_job_status()`: Get job status
- `get_job_result()`: Get job result
- `cancel_job()`: Cancel a job
- `list_jobs()`: List jobs
- `get_logs()`: Get logs
- `deploy_model()`: Deploy a model
- `delete_endpoint()`: Delete an endpoint
- `list_platforms()`: List available platforms
- `list_configured_platforms()`: List configured platforms
- `get_platform_info()`: Get platform information

### ResourceConfig

Configuration for compute resources.

**Attributes:**
- `instance_type`: Instance type (platform-specific)
- `gpu_enabled`: Whether GPU is enabled
- `gpu_count`: Number of GPUs
- `memory_gb`: Memory in GB
- `cpu_count`: Number of CPUs
- `disk_size_gb`: Disk size in GB
- `max_runtime_hours`: Maximum runtime in hours
- `auto_shutdown`: Enable auto-shutdown
- `custom_params`: Custom parameters

### JobStatus

Job status enumeration.

**Values:**
- `PENDING`: Job is pending
- `RUNNING`: Job is running
- `SUCCEEDED`: Job completed successfully
- `FAILED`: Job failed
- `CANCELLED`: Job was cancelled
- `UNKNOWN`: Status unknown

### JobResult

Result from a job execution.

**Attributes:**
- `job_id`: Job identifier
- `status`: Job status
- `output`: Job output
- `error`: Error message (if failed)
- `metrics`: Performance metrics
- `artifacts`: Output artifacts
- `logs_url`: URL to view logs
- `duration_seconds`: Execution duration
