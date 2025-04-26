"""
Unit tests for the Neural Cloud Integration module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.cloud.cloud_execution import CloudExecutor


class TestCloudExecutor(unittest.TestCase):
    """Test the CloudExecutor class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'NEURAL_CLOUD_ENV': 'test',
            'NEURAL_FORCE_CPU': '0'
        })
        self.env_patcher.start()

        # Create a mock for subprocess
        self.subprocess_patcher = patch('neural.cloud.cloud_execution.subprocess')
        self.mock_subprocess = self.subprocess_patcher.start()

        # Mock GPU availability
        self.mock_subprocess.run.return_value.returncode = 0  # Simulate GPU available

        # Create the executor with mocks
        with patch('neural.cloud.cloud_execution.NEURAL_IMPORTED', True):
            with patch('neural.cloud.cloud_execution.create_parser'):
                with patch('neural.cloud.cloud_execution.ModelTransformer'):
                    with patch('neural.cloud.cloud_execution.ShapePropagator'):
                        self.executor = CloudExecutor(environment='test')
                        self.executor.temp_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after the tests."""
        self.env_patcher.stop()
        self.subprocess_patcher.stop()
        self.temp_dir.cleanup()

    def test_detect_environment(self):
        """Test environment detection."""
        # Test Kaggle detection
        with patch.dict('os.environ', {'KAGGLE_KERNEL_RUN_TYPE': 'interactive'}):
            self.assertEqual(self.executor._detect_environment(), 'kaggle')

        # Test Colab detection
        with patch.dict('os.environ', {'COLAB_GPU': '1'}):
            self.assertEqual(self.executor._detect_environment(), 'colab')

        # Test SageMaker detection
        with patch.dict('os.environ', {'SM_MODEL_DIR': '/opt/ml/model'}):
            self.assertEqual(self.executor._detect_environment(), 'sagemaker')

        # Test unknown environment
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(self.executor._detect_environment(), 'unknown')

    def test_check_gpu_availability(self):
        """Test GPU availability detection."""
        # Test GPU available
        self.mock_subprocess.run.return_value.returncode = 0
        self.assertTrue(self.executor._check_gpu_availability())

        # Test GPU not available
        self.mock_subprocess.run.return_value.returncode = 1
        self.assertFalse(self.executor._check_gpu_availability())

        # Test nvidia-smi not found
        self.mock_subprocess.run.side_effect = FileNotFoundError
        self.assertFalse(self.executor._check_gpu_availability())

    @patch('neural.cloud.cloud_execution.NEURAL_IMPORTED', True)
    @patch('neural.cloud.cloud_execution.generate_code')
    def test_compile_model(self, mock_generate_code):
        """Test model compilation."""
        # Mock the parser and transformer
        self.executor.parser = MagicMock()
        self.executor.transformer = MagicMock()

        # Mock generate_code
        mock_generate_code.return_value = "# Generated code"

        # Test compilation
        dsl_code = "network TestModel { input: (10, 10, 3) }"
        output_path = self.executor.compile_model(dsl_code, backend='tensorflow')

        # Verify the parser and transformer were called
        self.executor.parser.parse.assert_called_once_with(dsl_code)
        self.executor.transformer.transform.assert_called_once()

        # Verify generate_code was called
        mock_generate_code.assert_called_once()

        # Verify the output file was created
        self.assertTrue(os.path.exists(output_path))

        # Verify the content of the output file
        with open(output_path, 'r') as f:
            self.assertEqual(f.read(), "# Generated code")

    def test_run_model(self):
        """Test model execution."""
        # Create a mock model file
        model_file = os.path.join(self.temp_dir.name, "model.py")
        with open(model_file, 'w') as f:
            f.write("# Test model")

        # Mock subprocess.run
        self.mock_subprocess.run.return_value.returncode = 0
        self.mock_subprocess.run.return_value.stdout = "Test output"
        self.mock_subprocess.run.return_value.stderr = ""

        # Test running the model
        result = self.executor.run_model(model_file, dataset='MNIST')

        # Verify subprocess.run was called
        self.mock_subprocess.run.assert_called_once()

        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['stdout'], "Test output")
        self.assertEqual(result['stderr'], "")

    @patch('neural.cloud.cloud_execution.NEURAL_IMPORTED', True)
    @patch('neural.cloud.cloud_execution.visualize_model')
    def test_visualize_model(self, mock_visualize_model):
        """Test model visualization."""
        # Mock the parser and transformer
        self.executor.parser = MagicMock()
        self.executor.transformer = MagicMock()

        # Test visualization
        dsl_code = "network TestModel { input: (10, 10, 3) }"
        output_path = self.executor.visualize_model(dsl_code, output_format='png')

        # Verify the parser and transformer were called
        self.executor.parser.parse.assert_called_once_with(dsl_code)
        self.executor.transformer.transform.assert_called_once()

        # Verify visualize_model was called
        mock_visualize_model.assert_called_once()

        # Verify the output path
        self.assertTrue(output_path.endswith('.png'))

    @patch('neural.cloud.cloud_execution.ngrok')
    def test_setup_ngrok_tunnel(self, mock_ngrok):
        """Test ngrok tunnel setup."""
        # Mock ngrok.connect
        mock_ngrok.connect.return_value.public_url = "https://test.ngrok.io"

        # Test tunnel setup
        url = self.executor.setup_ngrok_tunnel(port=8050)

        # Verify ngrok.connect was called
        mock_ngrok.connect.assert_called_once_with(8050)

        # Verify the URL
        self.assertEqual(url, "https://test.ngrok.io")

        # Test exception handling
        mock_ngrok.connect.side_effect = Exception("Test error")
        url = self.executor.setup_ngrok_tunnel(port=8050)
        self.assertIsNone(url)

    def test_start_debug_dashboard(self):
        """Test starting the debug dashboard."""
        # Mock subprocess.Popen
        mock_process = MagicMock()
        mock_process.pid = 12345
        self.mock_subprocess.Popen.return_value = mock_process

        # Mock setup_ngrok_tunnel
        self.executor.setup_ngrok_tunnel = MagicMock()
        self.executor.setup_ngrok_tunnel.return_value = "https://test.ngrok.io"

        # Test starting the dashboard
        dsl_code = "network TestModel { input: (10, 10, 3) }"
        result = self.executor.start_debug_dashboard(dsl_code, setup_tunnel=True)

        # Verify subprocess.Popen was called
        self.mock_subprocess.Popen.assert_called_once()

        # Verify setup_ngrok_tunnel was called
        self.executor.setup_ngrok_tunnel.assert_called_once_with(8050)

        # Verify the result
        self.assertEqual(result['session_id'], "debug_12345")
        self.assertEqual(result['dashboard_url'], "http://localhost:8050")
        self.assertEqual(result['process_id'], 12345)
        self.assertEqual(result['tunnel_url'], "https://test.ngrok.io")

    def test_start_nocode_interface(self):
        """Test starting the No-Code interface."""
        # Mock subprocess.Popen
        mock_process = MagicMock()
        mock_process.pid = 12345
        self.mock_subprocess.Popen.return_value = mock_process

        # Mock setup_ngrok_tunnel
        self.executor.setup_ngrok_tunnel = MagicMock()
        self.executor.setup_ngrok_tunnel.return_value = "https://test.ngrok.io"

        # Test starting the interface
        result = self.executor.start_nocode_interface(port=8051, setup_tunnel=True)

        # Verify subprocess.Popen was called
        self.mock_subprocess.Popen.assert_called_once()

        # Verify setup_ngrok_tunnel was called
        self.executor.setup_ngrok_tunnel.assert_called_once_with(8051)

        # Verify the result
        self.assertEqual(result['session_id'], "nocode_12345")
        self.assertEqual(result['interface_url'], "http://localhost:8051")
        self.assertEqual(result['process_id'], 12345)
        self.assertEqual(result['tunnel_url'], "https://test.ngrok.io")

    @patch('neural.cloud.cloud_execution.shutil')
    @patch('neural.cloud.cloud_execution.ngrok')
    def test_cleanup(self, mock_ngrok, mock_shutil):
        """Test cleanup."""
        # Test cleanup
        self.executor.cleanup()

        # Verify shutil.rmtree was called
        mock_shutil.rmtree.assert_called_once_with(self.executor.temp_dir)

        # Verify ngrok.kill was called
        mock_ngrok.kill.assert_called_once()

        # Test exception handling
        mock_shutil.rmtree.side_effect = Exception("Test error")
        self.executor.cleanup()  # Should not raise an exception


if __name__ == '__main__':
    unittest.main()
