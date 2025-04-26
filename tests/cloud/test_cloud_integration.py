"""
Integration tests for the Neural Cloud Integration module.

These tests are designed to be run in actual cloud environments.
They will be skipped if not running in the appropriate environment.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the cloud module
try:
    from neural.cloud.cloud_execution import CloudExecutor
    CLOUD_MODULE_AVAILABLE = True
except ImportError:
    CLOUD_MODULE_AVAILABLE = False


@unittest.skipIf(not CLOUD_MODULE_AVAILABLE, "Cloud module not available")
class TestCloudIntegration(unittest.TestCase):
    """Integration tests for cloud environments."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create the executor
        self.executor = CloudExecutor()
        self.executor.temp_dir = Path(self.temp_dir.name)

        # Test DSL code
        self.dsl_code = """
        network TestModel {
            input: (28, 28, 1)
            layers:
                Conv2D(32, (3, 3), "relu")
                MaxPooling2D((2, 2))
                Flatten()
                Dense(128, "relu")
                Dense(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
        }
        """

    def tearDown(self):
        """Clean up after the tests."""
        self.temp_dir.cleanup()
        self.executor.cleanup()

    @unittest.skipIf('KAGGLE_KERNEL_RUN_TYPE' not in os.environ, "Not running in Kaggle")
    def test_kaggle_environment(self):
        """Test running in Kaggle environment."""
        # Verify environment detection
        self.assertEqual(self.executor.environment, 'kaggle')

        # Test compilation
        model_path = self.executor.compile_model(self.dsl_code, backend='tensorflow')
        self.assertTrue(os.path.exists(model_path))

        # Test visualization
        viz_path = self.executor.visualize_model(self.dsl_code, output_format='png')
        self.assertTrue(os.path.exists(viz_path))

    @unittest.skipIf('COLAB_GPU' not in os.environ, "Not running in Colab")
    def test_colab_environment(self):
        """Test running in Colab environment."""
        # Verify environment detection
        self.assertEqual(self.executor.environment, 'colab')

        # Test compilation
        model_path = self.executor.compile_model(self.dsl_code, backend='pytorch')
        self.assertTrue(os.path.exists(model_path))

        # Test visualization
        viz_path = self.executor.visualize_model(self.dsl_code, output_format='svg')
        self.assertTrue(os.path.exists(viz_path))

    @unittest.skipIf('SM_MODEL_DIR' not in os.environ, "Not running in SageMaker")
    def test_sagemaker_environment(self):
        """Test running in SageMaker environment."""
        # Verify environment detection
        self.assertEqual(self.executor.environment, 'sagemaker')

        # Test compilation
        model_path = self.executor.compile_model(self.dsl_code, backend='tensorflow')
        self.assertTrue(os.path.exists(model_path))

        # Test model execution
        result = self.executor.run_model(model_path, dataset='MNIST')
        self.assertTrue(result['success'])


@unittest.skipIf(not CLOUD_MODULE_AVAILABLE, "Cloud module not available")
class TestRemoteConnection(unittest.TestCase):
    """Test remote connection to cloud environments."""

    def setUp(self):
        """Set up the test environment."""
        # Import the remote connection module
        try:
            from neural.cloud.remote_connection import RemoteConnection
            self.remote_connection_available = True
            self.remote = RemoteConnection()
        except ImportError:
            self.remote_connection_available = False

    @unittest.skipIf(not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')),
                    "Kaggle credentials not found")
    def test_kaggle_connection(self):
        """Test connection to Kaggle."""
        if not self.remote_connection_available:
            self.skipTest("Remote connection module not available")

        # Test connection
        result = self.remote.connect_to_kaggle()
        self.assertTrue(result['success'])

        # Test kernel creation
        kernel = self.remote.create_kaggle_kernel("test_neural_kernel")
        self.assertIsNotNone(kernel)

        # Test code execution
        response = self.remote.execute_on_kaggle(kernel, "print('Hello from Kaggle')")
        self.assertTrue(response['success'])

        # Clean up
        self.remote.delete_kaggle_kernel(kernel)

    @unittest.skipIf(not os.path.exists(os.path.expanduser('~/.aws/credentials')),
                    "AWS credentials not found")
    def test_sagemaker_connection(self):
        """Test connection to SageMaker."""
        if not self.remote_connection_available:
            self.skipTest("Remote connection module not available")

        # Test connection
        result = self.remote.connect_to_sagemaker()
        self.assertTrue(result['success'])

        # Test notebook instance creation
        instance = self.remote.create_sagemaker_notebook("test-neural-notebook")
        self.assertIsNotNone(instance)

        # Test code execution
        response = self.remote.execute_on_sagemaker(instance, "print('Hello from SageMaker')")
        self.assertTrue(response['success'])

        # Clean up
        self.remote.delete_sagemaker_notebook(instance)


if __name__ == '__main__':
    unittest.main()
