"""
Remote Connection Module for Neural DSL

This module provides functions to connect to cloud environments from a local terminal.
"""

import os
import sys
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging - redirect to file to avoid cluttering the console
log_file = os.path.join(tempfile.gettempdir(), "neural_cloud.log")
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.propagate = False  # Don't propagate to root logger (console)

class RemoteConnection:
    """Class for connecting to remote cloud environments."""

    def __init__(self):
        """Initialize the remote connection."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neural_remote_"))
        self.connections = {}

    def connect_to_kaggle(self) -> Dict[str, Any]:
        """
        Connect to Kaggle.

        Returns:
            Dictionary with connection information
        """
        try:
            # Check if kaggle CLI is installed
            try:
                subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Install kaggle CLI
                subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

            # Check if credentials are configured
            credentials_path = Path.home() / ".kaggle" / "kaggle.json"
            if not credentials_path.exists():
                return {
                    'success': False,
                    'error': "Kaggle credentials not found. Please run 'kaggle config' to set up your credentials."
                }

            # Test the connection
            result = subprocess.run(["kaggle", "kernels", "list"], check=True, capture_output=True, text=True)

            # Store the connection
            self.connections['kaggle'] = {
                'type': 'kaggle',
                'connected': True,
                'timestamp': time.time()
            }

            return {
                'success': True,
                'message': "Connected to Kaggle successfully"
            }
        except Exception as e:
            logger.error(f"Failed to connect to Kaggle: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def connect_to_colab(self) -> Dict[str, Any]:
        """
        Connect to Google Colab.

        Returns:
            Dictionary with connection information
        """
        try:
            # Check if gcloud CLI is installed
            try:
                subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return {
                    'success': False,
                    'error': "Google Cloud SDK not found. Please install it from https://cloud.google.com/sdk/docs/install"
                }

            # Check if user is authenticated
            result = subprocess.run(["gcloud", "auth", "list"], check=True, capture_output=True, text=True)
            if "No credentialed accounts." in result.stdout:
                return {
                    'success': False,
                    'error': "Not authenticated with Google Cloud. Please run 'gcloud auth login'"
                }

            # Store the connection
            self.connections['colab'] = {
                'type': 'colab',
                'connected': True,
                'timestamp': time.time()
            }

            return {
                'success': True,
                'message': "Connected to Google Cloud successfully"
            }
        except Exception as e:
            logger.error(f"Failed to connect to Colab: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def connect_to_sagemaker(self) -> Dict[str, Any]:
        """
        Connect to AWS SageMaker.

        Returns:
            Dictionary with connection information
        """
        try:
            # Check if AWS CLI is installed
            try:
                subprocess.run(["aws", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Install AWS CLI
                subprocess.run([sys.executable, "-m", "pip", "install", "awscli"], check=True)

            # Check if boto3 is installed
            try:
                import boto3
            except ImportError:
                # Install boto3
                subprocess.run([sys.executable, "-m", "pip", "install", "boto3"], check=True)
                import boto3

            # Check if credentials are configured
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials is None:
                    return {
                        'success': False,
                        'error': "AWS credentials not found. Please run 'aws configure' to set up your credentials."
                    }
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Failed to get AWS credentials: {e}"
                }

            # Test the connection
            sagemaker = boto3.client('sagemaker')
            sagemaker.list_notebook_instances()

            # Store the connection
            self.connections['sagemaker'] = {
                'type': 'sagemaker',
                'connected': True,
                'timestamp': time.time(),
                'session': session
            }

            return {
                'success': True,
                'message': "Connected to AWS SageMaker successfully"
            }
        except Exception as e:
            logger.error(f"Failed to connect to SageMaker: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_kaggle_kernel(self, kernel_name: str, code: Optional[str] = None,
                           dataset: Optional[str] = None) -> Optional[str]:
        """
        Create a Kaggle kernel.

        Args:
            kernel_name: Name of the kernel
            code: Code to run in the kernel
            dataset: Dataset to use

        Returns:
            Kernel ID if successful, None otherwise
        """
        try:
            # Check if connected to Kaggle
            if 'kaggle' not in self.connections or not self.connections['kaggle']['connected']:
                result = self.connect_to_kaggle()
                if not result['success']:
                    logger.error(f"Failed to connect to Kaggle: {result.get('error', 'Unknown error')}")
                    return None

            # Get Kaggle username
            kaggle_username = None
            try:
                # Try to get username from environment variable
                kaggle_username = os.environ.get('KAGGLE_USERNAME')

                # If not found, try to get it from the credentials file
                if not kaggle_username:
                    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
                    if credentials_path.exists():
                        with open(credentials_path, 'r') as f:
                            credentials = json.load(f)
                            kaggle_username = credentials.get('username')

                # If still not found, try to get it from the API
                if not kaggle_username:
                    result = subprocess.run(
                        ["kaggle", "config", "view"],
                        check=True, capture_output=True, text=True
                    )
                    for line in result.stdout.splitlines():
                        if "username" in line.lower():
                            kaggle_username = line.split(":")[-1].strip()
                            break

                # If still not found, use a default
                if not kaggle_username:
                    logger.warning("Could not determine Kaggle username. Using 'unknown'.")
                    kaggle_username = "unknown"

            except Exception as e:
                logger.warning(f"Failed to get Kaggle username: {e}. Using 'unknown'.")
                kaggle_username = "unknown"

            # Create a metadata file with a unique kernel name
            import uuid
            unique_suffix = uuid.uuid4().hex[:8]
            kernel_slug = f"{kernel_name.lower().replace(' ', '-')}-{unique_suffix}"

            metadata = {
                "id": f"{kaggle_username}/{kernel_slug}",
                "title": f"{kernel_name} {unique_suffix}",
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }

            if dataset:
                metadata["dataset_sources"].append(dataset)

            metadata_path = self.temp_dir / "kernel-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            # Create the code file
            code_path = self.temp_dir / "kernel.py"
            with open(code_path, 'w') as f:
                if code:
                    f.write(code)
                else:
                    # Default code to install Neural
                    f.write("""
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")
""")

            # Push the kernel to Kaggle
            try:
                logger.info(f"Pushing kernel to Kaggle with metadata: {metadata}")
                logger.info(f"Kernel directory: {self.temp_dir}")

                # List the directory contents for debugging
                logger.info(f"Directory contents: {os.listdir(self.temp_dir)}")

                # Run the command with more detailed output
                result = subprocess.run(
                    ["kaggle", "kernels", "push", "-p", str(self.temp_dir)],
                    capture_output=True, text=True
                )

                # Check for errors
                if result.returncode != 0:
                    logger.error(f"Kaggle API error: {result.stderr}")

                    # Try to provide more helpful error messages
                    if "not authorized" in result.stderr.lower() or "unauthorized" in result.stderr.lower():
                        logger.error("Authentication error. Please check your Kaggle API credentials.")
                        logger.error("Run 'kaggle config' to set up your credentials.")
                        return None
                    elif "already exists" in result.stderr.lower():
                        logger.error(f"A kernel with the name '{kernel_name}' already exists.")
                        # We're already using a unique name, so this shouldn't happen
                        return None
                    elif "invalid" in result.stderr.lower() and "metadata" in result.stderr.lower():
                        logger.error("Invalid metadata. Please check the kernel metadata format.")
                        return None
                    else:
                        # Try a simpler approach - just use the kernel name without username
                        logger.info("Trying a simpler approach...")
                        metadata["id"] = kernel_slug

                        # Write the updated metadata
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)

                        # Try again
                        result = subprocess.run(
                            ["kaggle", "kernels", "push", "-p", str(self.temp_dir)],
                            capture_output=True, text=True
                        )

                        if result.returncode != 0:
                            logger.error(f"Still failed: {result.stderr}")
                            return None

                logger.info(f"Kernel push successful: {result.stdout}")

            except Exception as e:
                logger.error(f"Failed to push kernel to Kaggle: {e}")
                return None

            # Extract the kernel ID
            kernel_id = metadata["id"]
            logger.info(f"Created Kaggle kernel: {kernel_id}")

            return kernel_id
        except Exception as e:
            logger.error(f"Failed to create Kaggle kernel: {e}")
            return None

    def execute_on_kaggle(self, kernel_id: str, code: str) -> Dict[str, Any]:
        """
        Execute code on a Kaggle kernel.

        Args:
            kernel_id: Kernel ID
            code: Code to execute

        Returns:
            Dictionary with execution results
        """
        try:
            # Check if connected to Kaggle
            if 'kaggle' not in self.connections or not self.connections['kaggle']['connected']:
                result = self.connect_to_kaggle()
                if not result['success']:
                    return {
                        'success': False,
                        'error': f"Failed to connect to Kaggle: {result.get('error', 'Unknown error')}"
                    }

            # Update the kernel code
            code_path = self.temp_dir / "kernel.py"
            with open(code_path, 'w') as f:
                f.write(code)

            # Push the updated kernel
            try:
                logger.info(f"Pushing updated kernel code to {kernel_id}")
                push_result = subprocess.run(
                    ["kaggle", "kernels", "push", "-p", str(self.temp_dir)],
                    capture_output=True, text=True
                )

                if push_result.returncode != 0:
                    logger.error(f"Failed to push kernel code: {push_result.stderr}")

                    # Try to recreate the kernel
                    logger.info("Trying to recreate the kernel...")
                    new_kernel_id = self.create_kaggle_kernel(f"neural-{int(time.time())}", code)
                    if not new_kernel_id:
                        return {
                            'success': False,
                            'error': f"Failed to push kernel code and couldn't recreate kernel: {push_result.stderr}"
                        }

                    logger.info(f"Created new kernel: {new_kernel_id}")
                    kernel_id = new_kernel_id
                else:
                    logger.info("Kernel code pushed successfully")

            except Exception as e:
                logger.error(f"Error pushing kernel code: {e}")
                return {
                    'success': False,
                    'error': f"Error pushing kernel code: {e}"
                }

            # The Kaggle API doesn't have a 'run' command
            # Instead, we'll just push the code and wait for it to execute
            try:
                logger.info(f"Waiting for kernel to execute: {kernel_id}")

                # The kernel should automatically run after being pushed
                # We'll just wait a bit to let it start
                time.sleep(5)

                logger.info("Kernel should be running now")

            except Exception as e:
                logger.error(f"Error waiting for kernel: {e}")
                return {
                    'success': False,
                    'error': f"Error waiting for kernel: {e}"
                }

            # Wait for the kernel to complete
            try:
                logger.info("Waiting for kernel to complete...")
                status = "unknown"

                # Try to check status, but don't fail if it doesn't work
                try:
                    # Wait a bit to let the kernel run
                    time.sleep(10)

                    # Check status
                    status_result = subprocess.run(
                        ["kaggle", "kernels", "status", kernel_id],
                        capture_output=True, text=True
                    )

                    if status_result.returncode == 0:
                        status = status_result.stdout.strip()
                        logger.info(f"Kernel status: {status}")
                    else:
                        logger.warning(f"Could not get kernel status: {status_result.stderr}")
                        # Continue anyway - we'll try to get the output
                except Exception as e:
                    logger.warning(f"Error checking kernel status: {e}")
                    # Continue anyway - we'll try to get the output

                logger.info("Proceeding to get kernel output")

            except Exception as e:
                logger.error(f"Error waiting for kernel: {e}")
                # Continue anyway - we'll try to get the output

            # Get the output
            try:
                logger.info("Getting kernel output...")

                # Try multiple times to get the output
                max_attempts = 3
                output = "No output available"

                for attempt in range(1, max_attempts + 1):
                    logger.info(f"Attempt {attempt}/{max_attempts} to get kernel output...")

                    try:
                        output_result = subprocess.run(
                            ["kaggle", "kernels", "output", kernel_id, "-p", str(self.temp_dir)],
                            capture_output=True, text=True
                        )

                        if output_result.returncode == 0:
                            # Read the output
                            output_path = self.temp_dir / "output.log"
                            if output_path.exists():
                                with open(output_path, 'r') as f:
                                    output = f.read()
                                logger.info(f"Got kernel output ({len(output)} characters)")
                                break  # Success, exit the loop
                            else:
                                logger.warning("No output file found")
                        else:
                            logger.warning(f"Failed to get kernel output: {output_result.stderr}")

                    except Exception as e:
                        logger.warning(f"Error getting kernel output: {e}")

                    # Wait before retrying
                    if attempt < max_attempts:
                        time.sleep(5)

                # Even if we couldn't get the output, return success
                # This allows the user to continue using the shell
                return {
                    'success': True,
                    'output': output,
                    'status': status
                }

            except Exception as e:
                logger.error(f"Error getting kernel output: {e}")
                return {
                    'success': False,
                    'error': f"Error getting kernel output: {e}"
                }

        except Exception as e:
            logger.error(f"Failed to execute on Kaggle: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def delete_kaggle_kernel(self, kernel_id: str) -> bool:
        """
        Delete a Kaggle kernel.

        Args:
            kernel_id: Kernel ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if connected to Kaggle
            if 'kaggle' not in self.connections or not self.connections['kaggle']['connected']:
                result = self.connect_to_kaggle()
                if not result['success']:
                    return False

            # Delete the kernel
            result = subprocess.run(
                ["kaggle", "kernels", "delete", kernel_id, "-y"],
                check=True, capture_output=True, text=True
            )

            return True
        except Exception as e:
            logger.error(f"Failed to delete Kaggle kernel: {e}")
            return False

    def create_sagemaker_notebook(self, notebook_name: str) -> Optional[str]:
        """
        Create a SageMaker notebook instance.

        Args:
            notebook_name: Name of the notebook instance

        Returns:
            Notebook instance name if successful, None otherwise
        """
        try:
            # Check if connected to SageMaker
            if 'sagemaker' not in self.connections or not self.connections['sagemaker']['connected']:
                result = self.connect_to_sagemaker()
                if not result['success']:
                    logger.error(f"Failed to connect to SageMaker: {result.get('error', 'Unknown error')}")
                    return None

            # Create the notebook instance
            import boto3
            sagemaker = boto3.client('sagemaker')

            # Get the execution role
            iam = boto3.client('iam')
            roles = iam.list_roles()
            role_arn = None
            for role in roles['Roles']:
                if 'SageMaker' in role['RoleName']:
                    role_arn = role['Arn']
                    break

            if not role_arn:
                logger.error("No SageMaker role found")
                return None

            # Create the notebook instance
            sagemaker.create_notebook_instance(
                NotebookInstanceName=notebook_name,
                InstanceType='ml.t2.medium',
                RoleArn=role_arn,
                LifecycleConfigName='',
                DirectInternetAccess='Enabled',
                VolumeSizeInGB=5,
                RootAccess='Enabled'
            )

            # Wait for the notebook instance to be in service
            waiter = sagemaker.get_waiter('notebook_instance_in_service')
            waiter.wait(NotebookInstanceName=notebook_name)

            return notebook_name
        except Exception as e:
            logger.error(f"Failed to create SageMaker notebook: {e}")
            return None

    def execute_on_sagemaker(self, notebook_name: str, code: str) -> Dict[str, Any]:
        """
        Execute code on a SageMaker notebook instance.

        Args:
            notebook_name: Notebook instance name
            code: Code to execute

        Returns:
            Dictionary with execution results
        """
        try:
            # Check if connected to SageMaker
            if 'sagemaker' not in self.connections or not self.connections['sagemaker']['connected']:
                result = self.connect_to_sagemaker()
                if not result['success']:
                    return {
                        'success': False,
                        'error': f"Failed to connect to SageMaker: {result.get('error', 'Unknown error')}"
                    }

            # Create a notebook file
            notebook_path = self.temp_dir / "notebook.ipynb"
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [code]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.10"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }

            with open(notebook_path, 'w') as f:
                json.dump(notebook, f)

            # Get the notebook instance URL
            import boto3
            sagemaker = boto3.client('sagemaker')
            response = sagemaker.describe_notebook_instance(NotebookInstanceName=notebook_name)
            notebook_url = response['Url']

            # Upload the notebook
            import requests
            upload_url = f"https://{notebook_url}/api/contents/notebook.ipynb"
            with open(notebook_path, 'r') as f:
                notebook_content = f.read()

            response = requests.put(
                upload_url,
                json={
                    "type": "notebook",
                    "content": json.loads(notebook_content)
                }
            )

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"Failed to upload notebook: {response.text}"
                }

            # Execute the notebook
            execute_url = f"https://{notebook_url}/api/kernels"
            response = requests.post(execute_url)

            if response.status_code != 201:
                return {
                    'success': False,
                    'error': f"Failed to start kernel: {response.text}"
                }

            kernel_id = response.json()['id']

            # Execute the code
            execute_url = f"https://{notebook_url}/api/kernels/{kernel_id}/execute"
            response = requests.post(
                execute_url,
                json={
                    "code": code
                }
            )

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"Failed to execute code: {response.text}"
                }

            msg_id = response.json()['msg_id']

            # Wait for execution to complete
            status_url = f"https://{notebook_url}/api/kernels/{kernel_id}/messages"
            output = []

            while True:
                response = requests.get(status_url)
                if response.status_code != 200:
                    return {
                        'success': False,
                        'error': f"Failed to get execution status: {response.text}"
                    }

                messages = response.json()
                for msg in messages:
                    if msg.get('parent_header', {}).get('msg_id') == msg_id:
                        if msg.get('msg_type') == 'stream':
                            output.append(msg.get('content', {}).get('text', ''))
                        elif msg.get('msg_type') == 'execute_result':
                            output.append(str(msg.get('content', {}).get('data', {}).get('text/plain', '')))
                        elif msg.get('msg_type') == 'error':
                            return {
                                'success': False,
                                'error': msg.get('content', {}).get('traceback', ['Unknown error'])[0]
                            }
                        elif msg.get('msg_type') == 'status' and msg.get('content', {}).get('execution_state') == 'idle':
                            return {
                                'success': True,
                                'output': ''.join(output)
                            }

                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to execute on SageMaker: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def delete_sagemaker_notebook(self, notebook_name: str) -> bool:
        """
        Delete a SageMaker notebook instance.

        Args:
            notebook_name: Notebook instance name

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if connected to SageMaker
            if 'sagemaker' not in self.connections or not self.connections['sagemaker']['connected']:
                result = self.connect_to_sagemaker()
                if not result['success']:
                    return False

            # Delete the notebook instance
            import boto3
            sagemaker = boto3.client('sagemaker')

            # Stop the notebook instance
            sagemaker.stop_notebook_instance(NotebookInstanceName=notebook_name)

            # Wait for the notebook instance to stop
            waiter = sagemaker.get_waiter('notebook_instance_stopped')
            waiter.wait(NotebookInstanceName=notebook_name)

            # Delete the notebook instance
            sagemaker.delete_notebook_instance(NotebookInstanceName=notebook_name)

            return True
        except Exception as e:
            logger.error(f"Failed to delete SageMaker notebook: {e}")
            return False

    def cleanup(self):
        """Clean up resources."""
        import shutil
        shutil.rmtree(self.temp_dir)
