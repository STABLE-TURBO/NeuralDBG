"""
AWS SageMaker Integration for Neural DSL

This module provides functions to run Neural DSL in AWS SageMaker environments.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger(__name__)

# Check if running in SageMaker
IS_SAGEMAKER = 'SM_MODEL_DIR' in os.environ

class SageMakerHandler:
    """Handler for AWS SageMaker integration."""

    def __init__(self):
        """Initialize the SageMaker handler."""
        self.model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        self.input_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
        self.output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
        self.hyperparameters = self._load_hyperparameters()

        # Set environment variables
        os.environ['NEURAL_CLOUD_ENV'] = 'sagemaker'

        # Create directories if they don't exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_hyperparameters(self) -> Dict[str, Any]:
        """Load hyperparameters from SageMaker."""
        hyperparameters_file = '/opt/ml/input/config/hyperparameters.json'
        if os.path.exists(hyperparameters_file):
            try:
                with open(hyperparameters_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load hyperparameters: {e}")
        return {}

    def load_dsl_from_s3(self, s3_uri: str) -> str:
        """
        Load a Neural DSL file from S3.

        Args:
            s3_uri: S3 URI of the DSL file

        Returns:
            The DSL code as a string
        """
        try:
            import boto3
            from urllib.parse import urlparse

            # Parse the S3 URI
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')

            # Download the file
            s3 = boto3.client('s3')
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                s3.download_file(bucket, key, tmp.name)
                with open(tmp.name, 'r') as f:
                    dsl_code = f.read()
                os.unlink(tmp.name)

            return dsl_code
        except Exception as e:
            logger.error(f"Failed to load DSL from S3: {e}")
            raise

    def save_model_to_s3(self, model_path: str, s3_uri: str) -> bool:
        """
        Save a model to S3.

        Args:
            model_path: Path to the model file
            s3_uri: S3 URI to save the model to

        Returns:
            True if successful, False otherwise
        """
        try:
            import boto3
            from urllib.parse import urlparse

            # Parse the S3 URI
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')

            # Upload the file
            s3 = boto3.client('s3')
            s3.upload_file(model_path, bucket, key)

            return True
        except Exception as e:
            logger.error(f"Failed to save model to S3: {e}")
            return False

    def train_model(self, dsl_code: str, backend: str = 'tensorflow') -> Dict[str, Any]:
        """
        Train a model in SageMaker.

        Args:
            dsl_code: Neural DSL code
            backend: Backend to use (tensorflow, pytorch)

        Returns:
            Dictionary with training results
        """
        try:
            # Import Neural modules
            from neural.cloud.cloud_execution import CloudExecutor

            # Initialize the cloud executor
            executor = CloudExecutor(environment='sagemaker')

            # Compile the model
            model_path = executor.compile_model(dsl_code, backend=backend)

            # Run the model
            results = executor.run_model(
                model_path,
                dataset=self.hyperparameters.get('dataset', 'MNIST'),
                epochs=int(self.hyperparameters.get('epochs', 5)),
                batch_size=int(self.hyperparameters.get('batch_size', 32))
            )

            # Save the model
            model_output_path = os.path.join(self.model_dir, f"model_{backend}.py")
            import shutil
            shutil.copy(model_path, model_output_path)

            # Save the results
            results_path = os.path.join(self.output_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f)

            return {
                'success': results['success'],
                'model_path': model_output_path,
                'results_path': results_path
            }
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def deploy_model(self, model_path: str, endpoint_name: str) -> Dict[str, Any]:
        """
        Deploy a model to a SageMaker endpoint.

        Args:
            model_path: Path to the model file
            endpoint_name: Name of the endpoint

        Returns:
            Dictionary with deployment results
        """
        try:
            import boto3

            # Create a model.tar.gz file
            model_dir = tempfile.mkdtemp()
            model_tar_path = os.path.join(model_dir, 'model.tar.gz')

            # Copy the model file
            import shutil
            import tarfile

            # Create the inference script
            inference_script = os.path.join(model_dir, 'inference.py')
            with open(inference_script, 'w') as f:
                f.write(self._generate_inference_script())

            # Create the tar.gz file
            with tarfile.open(model_tar_path, 'w:gz') as tar:
                tar.add(model_path, arcname=os.path.basename(model_path))
                tar.add(inference_script, arcname='inference.py')

            # Upload the model to S3
            sagemaker = boto3.client('sagemaker')
            s3 = boto3.client('s3')

            # Create a unique bucket name
            import uuid
            bucket_name = f"neural-sagemaker-{uuid.uuid4()}"
            s3.create_bucket(Bucket=bucket_name)

            # Upload the model
            model_s3_key = 'model/model.tar.gz'
            s3.upload_file(model_tar_path, bucket_name, model_s3_key)
            model_s3_uri = f"s3://{bucket_name}/{model_s3_key}"

            # Create the model in SageMaker
            role = os.environ.get('SAGEMAKER_ROLE', None)
            if not role:
                # Try to get the role from the instance
                import requests
                r = requests.get('http://169.254.169.254/latest/meta-data/iam/info')
                if r.status_code == 200:
                    role = r.json()['InstanceProfileArn']
                else:
                    raise ValueError("SageMaker role not found")

            # Create the model
            model_name = f"neural-model-{uuid.uuid4()}"
            sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8.0-cpu',
                    'ModelDataUrl': model_s3_uri,
                    'Environment': {
                        'NEURAL_CLOUD_ENV': 'sagemaker',
                        'NEURAL_MODEL_PATH': os.path.basename(model_path)
                    }
                },
                ExecutionRoleArn=role
            )

            # Create an endpoint configuration
            endpoint_config_name = f"neural-endpoint-config-{uuid.uuid4()}"
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.t2.medium'
                    }
                ]
            )

            # Create the endpoint
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )

            # Clean up
            shutil.rmtree(model_dir)

            return {
                'success': True,
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'model_s3_uri': model_s3_uri
            }
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_inference_script(self) -> str:
        """Generate an inference script for SageMaker."""
        return """
import os
import sys
import json
import importlib.util
import numpy as np

# Path to the model file
model_path = os.environ.get('NEURAL_MODEL_PATH', 'model.py')

# Load the model module
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Get the model
model = model_module.model

def model_fn(model_dir):
    # Return the loaded model
    return model

def input_fn(request_body, request_content_type):
    # Parse the input
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Make a prediction
    return model.predict(input_data)

def output_fn(prediction, accept):
    # Format the output
    if accept == 'application/json':
        return json.dumps(prediction.tolist()), 'application/json'
    else:
        return json.dumps(prediction.tolist()), 'application/json'
"""


def sagemaker_entry_point():
    """Entry point for SageMaker training."""
    if not IS_SAGEMAKER:
        print("Not running in SageMaker environment")
        return

    # Initialize the handler
    handler = SageMakerHandler()

    # Get the DSL code
    dsl_path = os.path.join(handler.input_dir, 'model.neural')
    if os.path.exists(dsl_path):
        with open(dsl_path, 'r') as f:
            dsl_code = f.read()
    else:
        # Try to get the DSL code from hyperparameters
        s3_uri = handler.hyperparameters.get('dsl_s3_uri')
        if s3_uri:
            dsl_code = handler.load_dsl_from_s3(s3_uri)
        else:
            raise ValueError("DSL code not found")

    # Train the model
    backend = handler.hyperparameters.get('backend', 'tensorflow')
    result = handler.train_model(dsl_code, backend=backend)

    if result['success']:
        print(f"Model trained successfully: {result['model_path']}")
    else:
        print(f"Model training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    sagemaker_entry_point()
