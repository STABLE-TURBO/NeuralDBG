"""
Unit tests for the Neural Cloud Notebook Interface.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path
import json

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.cloud.notebook_interface import NeuralNotebook


class TestNotebookInterface(unittest.TestCase):
    """Test the NeuralNotebook class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock the RemoteConnection class
        self.remote_patcher = patch('neural.cloud.notebook_interface.RemoteConnection')
        self.mock_remote_class = self.remote_patcher.start()
        self.mock_remote = MagicMock()
        self.mock_remote_class.return_value = self.mock_remote

        # Mock the connect_to_kaggle method
        self.mock_remote.connect_to_kaggle.return_value = {'success': True, 'message': 'Connected to Kaggle'}

        # Mock the create_kaggle_kernel method
        self.mock_remote.create_kaggle_kernel.return_value = 'test-kernel-id'

        # Mock the execute_on_kaggle method
        self.mock_remote.execute_on_kaggle.return_value = {'success': True, 'output': 'Test output'}

        # Mock the subprocess module
        self.subprocess_patcher = patch('neural.cloud.notebook_interface.subprocess')
        self.mock_subprocess = self.subprocess_patcher.start()

        # Mock the webbrowser module
        self.webbrowser_patcher = patch('neural.cloud.notebook_interface.webbrowser')
        self.mock_webbrowser = self.webbrowser_patcher.start()

        # Create a notebook with mocks
        with patch('neural.cloud.notebook_interface.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = self.temp_dir.name
            with patch('builtins.open', mock_open()) as mock_file:
                self.notebook = NeuralNotebook('kaggle', self.mock_remote, port=8888)
                self.notebook.temp_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after the tests."""
        self.remote_patcher.stop()
        self.subprocess_patcher.stop()
        self.webbrowser_patcher.stop()
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test notebook initialization."""
        # Verify the remote connection was initialized
        self.assertEqual(self.notebook.platform, 'kaggle')
        self.assertEqual(self.notebook.remote, self.mock_remote)
        self.assertEqual(self.notebook.kernel_id, 'test-kernel-id')
        self.assertEqual(self.notebook.port, 8888)

        # Verify connect_to_kaggle was called
        self.mock_remote.connect_to_kaggle.assert_called_once()

        # Verify create_kaggle_kernel was called
        self.mock_remote.create_kaggle_kernel.assert_called_once()

        # Verify execute_on_kaggle was called for initialization
        self.mock_remote.execute_on_kaggle.assert_called_once()

    def test_create_notebook_file(self):
        """Test creating the notebook file."""
        # Create a real notebook file
        with patch('builtins.open', mock_open()) as mock_file:
            self.notebook._create_notebook_file()

        # Verify open was called
        mock_file.assert_called_once()

        # Verify the file path
        args, kwargs = mock_file.call_args
        self.assertEqual(args[0], self.notebook.notebook_path)
        self.assertEqual(kwargs['mode'], 'w')

        # Verify json.dump was called
        handle = mock_file()
        handle.write.assert_called_once()

    def test_start_notebook_server(self):
        """Test starting the notebook server."""
        # Mock the subprocess.run method
        self.mock_subprocess.run.return_value = MagicMock()

        # Mock the subprocess.Popen method
        self.mock_subprocess.Popen.return_value = MagicMock()

        # Mock the time.sleep method
        with patch('neural.cloud.notebook_interface.time.sleep') as mock_sleep:
            # Mock the setup_cell_execution_proxy method
            with patch.object(self.notebook, '_setup_cell_execution_proxy') as mock_setup:
                # Start the notebook server
                result = self.notebook.start_notebook_server()

        # Verify the result
        self.assertTrue(result)

        # Verify subprocess.run was called
        self.mock_subprocess.run.assert_called_once()

        # Verify subprocess.Popen was called
        self.mock_subprocess.Popen.assert_called_once()

        # Verify webbrowser.open was called
        self.mock_webbrowser.open.assert_called_once_with('http://localhost:8888/notebooks/neural_notebook.ipynb')

        # Verify setup_cell_execution_proxy was called
        mock_setup.assert_called_once()

    def test_execute_cell(self):
        """Test executing a cell."""
        # Execute a cell
        result = self.notebook.execute_cell('print("Hello, world!")')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_with('test-kernel-id', 'print("Hello, world!")')

        # Verify the result
        self.assertEqual(result, {'success': True, 'output': 'Test output'})

    def test_stop_notebook_server(self):
        """Test stopping the notebook server."""
        # Create a mock server process
        mock_process = MagicMock()
        self.notebook.server_process = mock_process

        # Stop the notebook server
        self.notebook.stop_notebook_server()

        # Verify terminate was called
        mock_process.terminate.assert_called_once()

        # Verify wait was called
        mock_process.wait.assert_called_once()

        # Verify the server process was set to None
        self.assertIsNone(self.notebook.server_process)

    def test_cleanup(self):
        """Test cleanup."""
        # Create a mock server process
        mock_process = MagicMock()
        self.notebook.server_process = mock_process

        # Mock the shutil module
        with patch('neural.cloud.notebook_interface.shutil') as mock_shutil:
            # Clean up
            self.notebook.cleanup()

        # Verify terminate was called
        mock_process.terminate.assert_called_once()

        # Verify delete_kaggle_kernel was called
        self.mock_remote.delete_kaggle_kernel.assert_called_once_with('test-kernel-id')

        # Verify cleanup was called
        self.mock_remote.cleanup.assert_called_once()

        # Verify shutil.rmtree was called
        mock_shutil.rmtree.assert_called_once_with(self.notebook.temp_dir)


if __name__ == '__main__':
    unittest.main()
