"""
Example usage of Neural DSL platform integrations.

Demonstrates common workflows and patterns.
"""

from neural.integrations import (
    AzureMLConnector,
    DatabricksConnector,
    JobStatus,
    PaperspaceConnector,
    PlatformManager,
    ResourceConfig,
    RunAIConnector,
    SageMakerConnector,
    VertexAIConnector,
)


def example_unified_manager():
    """Example using the unified PlatformManager."""
    print("=== Unified Manager Example ===\n")
    
    manager = PlatformManager()
    
    manager.configure_platform(
        'databricks',
        host='https://your-workspace.cloud.databricks.com',
        token='your-token'
    )
    
    dsl_code = """
model ImageClassifier {
    input: (None, 224, 224, 3)
    
    Conv2D(filters=32, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    
    Conv2D(filters=64, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    
    Flatten()
    Dense(units=128, activation='relu')
    Dense(units=10, activation='softmax')
}
"""
    
    resource_config = ResourceConfig(
        instance_type='i3.xlarge',
        gpu_enabled=False,
        memory_gb=32,
        auto_shutdown=True
    )
    
    job_id = manager.submit_job(
        platform='databricks',
        code=dsl_code,
        resource_config=resource_config,
        job_name='image-classifier-training'
    )
    
    print(f"Job submitted: {job_id}")
    
    status = manager.get_job_status('databricks', job_id)
    print(f"Job status: {status}")
    
    if status == JobStatus.SUCCEEDED:
        result = manager.get_job_result('databricks', job_id)
        print(f"Duration: {result.duration_seconds}s")
        print(f"Output: {result.output}")


def example_databricks():
    """Example using Databricks connector."""
    print("\n=== Databricks Example ===\n")
    
    connector = DatabricksConnector(credentials={
        'host': 'https://your-workspace.cloud.databricks.com',
        'token': 'dapi...',
        'cluster_id': '1234-567890-abc123'
    })
    
    connector.authenticate()
    
    code = """
from neural.parser.parser import parse_dsl
from neural.code_generation.tensorflow_generator import TensorFlowGenerator

dsl_code = '''
model SimpleNet {
    input: (None, 28, 28, 1)
    Flatten()
    Dense(units=128, activation='relu')
    Dense(units=10, activation='softmax')
}
'''

ast = parse_dsl(dsl_code)
generator = TensorFlowGenerator()
model_code = generator.generate(ast)

print("Generated model code:")
print(model_code)
"""
    
    job_id = connector.submit_job(
        code=code,
        resource_config=ResourceConfig(instance_type='i3.xlarge', gpu_enabled=False),
        job_name='neural-dsl-test'
    )
    
    print(f"Job submitted: {job_id}")
    
    jobs = connector.list_jobs(limit=5)
    for job in jobs:
        print(f"Job {job['job_id']}: {job['status']}")


def example_sagemaker():
    """Example using AWS SageMaker connector."""
    print("\n=== SageMaker Example ===\n")
    
    connector = SageMakerConnector(credentials={
        'access_key_id': 'AKIA...',
        'secret_access_key': 'wJalr...',
        'region': 'us-east-1',
        'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole',
        's3_bucket': 'my-sagemaker-bucket'
    })
    
    connector.authenticate()
    
    training_code = """
import tensorflow as tf
from neural.parser.parser import parse_dsl
from neural.code_generation.tensorflow_generator import TensorFlowGenerator

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Parse DSL and generate model
dsl_code = '''
model MNISTClassifier {
    input: (None, 28, 28)
    Reshape(target_shape=(28, 28, 1))
    Conv2D(filters=32, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=128, activation='relu')
    Dropout(rate=0.2)
    Dense(units=10, activation='softmax')
}
'''

ast = parse_dsl(dsl_code)
generator = TensorFlowGenerator()
exec(generator.generate(ast))

# Train model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

print(f"Test accuracy: {history.history['val_accuracy'][-1]}")
"""
    
    job_id = connector.submit_job(
        code=training_code,
        resource_config=ResourceConfig(
            instance_type='ml.p3.2xlarge',
            gpu_enabled=True,
            max_runtime_hours=2
        ),
        dependencies=['tensorflow>=2.12', 'neural-dsl'],
        job_name='mnist-training'
    )
    
    print(f"Training job submitted: {job_id}")


def example_vertex_ai():
    """Example using Google Vertex AI connector."""
    print("\n=== Vertex AI Example ===\n")
    
    connector = VertexAIConnector(credentials={
        'project_id': 'my-gcp-project',
        'location': 'us-central1'
    })
    
    connector.authenticate()
    
    job_id = connector.submit_job(
        code="""
import tensorflow as tf
from neural.parser.parser import parse_dsl
print("Training on Vertex AI...")
""",
        resource_config=ResourceConfig(
            instance_type='n1-standard-4',
            gpu_enabled=True,
            gpu_count=1
        ),
        job_name='vertex-training'
    )
    
    print(f"Job ID: {job_id}")


def example_azure_ml():
    """Example using Azure ML connector."""
    print("\n=== Azure ML Example ===\n")
    
    connector = AzureMLConnector(credentials={
        'subscription_id': '12345678-1234-1234-1234-123456789012',
        'resource_group': 'my-resource-group',
        'workspace_name': 'my-workspace'
    })
    
    connector.authenticate()
    
    job_id = connector.submit_job(
        code="""
import tensorflow as tf
print("Training on Azure ML...")
""",
        resource_config=ResourceConfig(
            instance_type='Standard_NC6',
            gpu_enabled=True
        ),
        job_name='azure-training'
    )
    
    print(f"Job ID: {job_id}")


def example_paperspace():
    """Example using Paperspace Gradient connector."""
    print("\n=== Paperspace Example ===\n")
    
    connector = PaperspaceConnector(credentials={
        'api_key': 'ps_...',
        'project_id': 'prj...'
    })
    
    connector.authenticate()
    
    job_id = connector.submit_job(
        code="""
import tensorflow as tf
print("Training on Paperspace Gradient...")
""",
        resource_config=ResourceConfig(
            instance_type='P4000',
            gpu_enabled=True
        ),
        job_name='gradient-training'
    )
    
    print(f"Job ID: {job_id}")


def example_runai():
    """Example using Run:AI connector."""
    print("\n=== Run:AI Example ===\n")
    
    connector = RunAIConnector(credentials={
        'project': 'my-project'
    })
    
    connector.authenticate()
    
    job_id = connector.submit_job(
        code="""
import tensorflow as tf
print("Distributed training on Run:AI...")
""",
        resource_config=ResourceConfig(
            instance_type='V100',
            gpu_enabled=True,
            gpu_count=4
        ),
        job_name='distributed-training'
    )
    
    print(f"Job ID: {job_id}")


def example_multi_platform():
    """Example submitting to multiple platforms."""
    print("\n=== Multi-Platform Example ===\n")
    
    manager = PlatformManager()
    
    manager.configure_platform('databricks', host='...', token='...')
    manager.configure_platform('sagemaker', access_key_id='...', secret_access_key='...')
    
    code = """
model MyModel {
    input: (None, 28, 28, 1)
    Conv2D(filters=32, kernel_size=3, activation='relu')
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=10, activation='softmax')
}
"""
    
    databricks_job = manager.submit_job('databricks', code=code)
    sagemaker_job = manager.submit_job('sagemaker', code=code)
    
    print(f"Databricks job: {databricks_job}")
    print(f"SageMaker job: {sagemaker_job}")
    
    print("\nJob statuses:")
    print(f"Databricks: {manager.get_job_status('databricks', databricks_job)}")
    print(f"SageMaker: {manager.get_job_status('sagemaker', sagemaker_job)}")


def example_model_deployment():
    """Example deploying a model."""
    print("\n=== Model Deployment Example ===\n")
    
    manager = PlatformManager()
    
    manager.configure_platform('databricks', host='...', token='...')
    
    endpoint_url = manager.deploy_model(
        platform='databricks',
        model_path='dbfs:/models/my-neural-model',
        endpoint_name='neural-model-prod',
        resource_config=ResourceConfig(
            instance_type='i3.large',
            gpu_enabled=False
        )
    )
    
    print(f"Model deployed to: {endpoint_url}")
    
    print("\nList configured platforms:")
    for platform in manager.list_configured_platforms():
        info = manager.get_platform_info(platform)
        print(f"  {platform}: {info['description']}")


def example_resource_management():
    """Example of resource management."""
    print("\n=== Resource Management Example ===\n")
    
    manager = PlatformManager()
    manager.configure_platform('runai', project='my-project')
    
    usage = manager.get_resource_usage('runai')
    print(f"Resource usage:")
    print(f"  Total GPUs: {usage.get('total_gpus', 'N/A')}")
    print(f"  Used GPUs: {usage.get('used_gpus', 'N/A')}")
    print(f"  Available GPUs: {usage.get('available_gpus', 'N/A')}")


def example_batch_processing():
    """Example of batch job submission."""
    print("\n=== Batch Processing Example ===\n")
    
    from neural.integrations.utils import batch_submit_jobs
    
    manager = PlatformManager()
    manager.configure_platform('databricks', host='...', token='...')
    
    jobs = [
        {
            'code': 'print("Job 1")',
            'job_name': 'batch-job-1',
            'resource_config': ResourceConfig(instance_type='i3.xlarge', gpu_enabled=False)
        },
        {
            'code': 'print("Job 2")',
            'job_name': 'batch-job-2',
            'resource_config': ResourceConfig(instance_type='i3.xlarge', gpu_enabled=False)
        },
        {
            'code': 'print("Job 3")',
            'job_name': 'batch-job-3',
            'resource_config': ResourceConfig(instance_type='i3.xlarge', gpu_enabled=False)
        },
    ]
    
    job_ids = batch_submit_jobs(manager, 'databricks', jobs)
    print(f"Submitted {len(job_ids)} jobs: {job_ids}")


if __name__ == '__main__':
    print("Neural DSL Platform Integrations Examples")
    print("=" * 50)
    print("\nNote: These are demonstration examples.")
    print("Replace placeholder credentials with actual values.\n")
    
    try:
        example_unified_manager()
    except Exception as e:
        print(f"Unified manager example error: {e}")
