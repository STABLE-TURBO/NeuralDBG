# Neural DSL Integrations - Quick Reference

## Installation

```bash
pip install neural-dsl[integrations]
```

## Basic Usage

```python
from neural.integrations import PlatformManager, ResourceConfig

manager = PlatformManager()
manager.configure_platform('databricks', host='...', token='...')
job_id = manager.submit_job('databricks', code=dsl_code)
status = manager.get_job_status('databricks', job_id)
```

## Platform Configuration

### Databricks
```python
manager.configure_platform('databricks',
    host='https://your-workspace.cloud.databricks.com',
    token='dapi...',
    cluster_id='optional-cluster-id'
)
```

### AWS SageMaker
```python
manager.configure_platform('sagemaker',
    access_key_id='AKIA...',
    secret_access_key='wJalr...',
    region='us-east-1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    s3_bucket='my-bucket'
)
```

### Google Vertex AI
```python
manager.configure_platform('vertex_ai',
    project_id='my-project',
    location='us-central1',
    credentials_file='/path/to/credentials.json'
)
```

### Azure ML
```python
manager.configure_platform('azure_ml',
    subscription_id='12345678-1234-1234-1234-123456789012',
    resource_group='my-rg',
    workspace_name='my-workspace'
)
```

### Paperspace
```python
manager.configure_platform('paperspace',
    api_key='ps_...',
    project_id='prj...'
)
```

### Run:AI
```python
manager.configure_platform('runai',
    cluster_url='https://my-cluster.run.ai',
    project='my-project'
)
```

## Resource Configuration

```python
from neural.integrations import ResourceConfig

# CPU configuration
cpu_config = ResourceConfig(
    instance_type='n1-standard-4',
    gpu_enabled=False,
    memory_gb=16,
    cpu_count=4
)

# GPU configuration
gpu_config = ResourceConfig(
    instance_type='p3.2xlarge',
    gpu_enabled=True,
    gpu_count=1,
    max_runtime_hours=24,
    auto_shutdown=True
)
```

## Job Management

### Submit Job
```python
job_id = manager.submit_job(
    platform='databricks',
    code=dsl_code,
    resource_config=gpu_config,
    environment={'ENV_VAR': 'value'},
    dependencies=['tensorflow>=2.12'],
    job_name='my-job'
)
```

### Check Status
```python
from neural.integrations import JobStatus

status = manager.get_job_status('databricks', job_id)
if status == JobStatus.SUCCEEDED:
    result = manager.get_job_result('databricks', job_id)
    print(result.output)
```

### List Jobs
```python
jobs = manager.list_jobs('databricks', limit=10)
for job in jobs:
    print(f"{job['job_id']}: {job['status']}")
```

### Get Logs
```python
logs = manager.get_logs('databricks', job_id)
print(logs)
```

### Cancel Job
```python
success = manager.cancel_job('databricks', job_id)
```

## Model Deployment

```python
endpoint = manager.deploy_model(
    platform='databricks',
    model_path='dbfs:/models/my-model',
    endpoint_name='neural-model',
    resource_config=ResourceConfig(
        instance_type='i3.large',
        gpu_enabled=False
    )
)

# Delete endpoint
manager.delete_endpoint('databricks', 'neural-model')
```

## Utility Functions

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

### Wait for Completion
```python
from neural.integrations.utils import wait_for_job_completion

result = wait_for_job_completion(
    manager,
    platform='databricks',
    job_id=job_id,
    poll_interval=30,
    timeout=3600
)
```

### Batch Submit
```python
from neural.integrations.utils import batch_submit_jobs

jobs = [
    {'code': code1, 'job_name': 'job-1'},
    {'code': code2, 'job_name': 'job-2'},
]

job_ids = batch_submit_jobs(manager, 'databricks', jobs)
```

## Environment Variables

### Databricks
```bash
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
```

### AWS
```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=wJalr...
export AWS_DEFAULT_REGION=us-east-1
```

### GCP
```bash
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Azure
```bash
export AZURE_SUBSCRIPTION_ID=12345678-1234-1234-1234-123456789012
export AZURE_RESOURCE_GROUP=my-rg
export AZURE_WORKSPACE_NAME=my-workspace
```

### Paperspace
```bash
export PAPERSPACE_API_KEY=ps_...
```

### Run:AI
```bash
export RUNAI_PROJECT=my-project
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

try:
    result = manager.get_job_result('databricks', job_id)
except CloudException as e:
    print(f"Error: {e}")
```

## Common Instance Types

### AWS SageMaker
- CPU: `ml.t2.medium`, `ml.m5.large`, `ml.m5.xlarge`
- GPU: `ml.p3.2xlarge`, `ml.p3.8xlarge`, `ml.g4dn.xlarge`

### Databricks
- CPU: `i3.xlarge`, `i3.2xlarge`, `r5.large`
- GPU: `p3.2xlarge`, `p3.8xlarge`, `g4dn.xlarge`

### GCP Vertex AI
- CPU: `n1-standard-4`, `n1-standard-8`, `n1-highmem-8`
- GPU: `n1-standard-4` + `NVIDIA_TESLA_T4`

### Azure ML
- CPU: `Standard_D2_v2`, `Standard_D4_v2`, `Standard_DS3_v2`
- GPU: `Standard_NC6`, `Standard_NC12`, `Standard_NC24`

### Paperspace
- CPU: `C5`, `C7`
- GPU: `P4000`, `P5000`, `V100`, `A100`

### Run:AI
- GPU: `V100`, `A100`, `T4`

## Best Practices

1. Always use `auto_shutdown=True` to save costs
2. Set `max_runtime_hours` to prevent runaway jobs
3. Use environment variables for credentials
4. Estimate costs before submitting expensive jobs
5. Monitor job status regularly
6. Clean up resources after use
7. Use appropriate instance types for workload
8. Enable GPU only when needed
9. Test with smaller instances first
10. Implement error handling and retries

## Platform Limits

| Platform | Max GPUs/Job | Max Duration | Auto-scaling |
|----------|--------------|--------------|--------------|
| Databricks | 8 | Unlimited | Yes |
| SageMaker | 8 | 5 days | Yes |
| Vertex AI | 8 | 7 days | Yes |
| Azure ML | 4 | 7 days | Yes |
| Paperspace | 8 | Unlimited | Limited |
| Run:AI | 16+ | Custom | Yes |

## Support

- Documentation: See `neural/integrations/README.md`
- Examples: See `neural/integrations/examples.py`
- Tests: See `tests/test_integrations.py`
- Issues: https://github.com/Lemniscate-world/Neural/issues
