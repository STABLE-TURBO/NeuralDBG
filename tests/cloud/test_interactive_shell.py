"""
Unit tests for the Neural Cloud Interactive Shell.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.cloud.interactive_shell import NeuralCloudShell


class TestInteractiveShell(unittest.TestCase):
    """Test the NeuralCloudShell class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock the RemoteConnection class
        self.remote_patcher = patch('neural.cloud.interactive_shell.RemoteConnection')
        self.mock_remote_class = self.remote_patcher.start()
        self.mock_remote = MagicMock()
        self.mock_remote_class.return_value = self.mock_remote

        # Mock the connect_to_kaggle method
        self.mock_remote.connect_to_kaggle.return_value = {'success': True, 'message': 'Connected to Kaggle'}

        # Mock the create_kaggle_kernel method
        self.mock_remote.create_kaggle_kernel.return_value = 'test-kernel-id'

        # Mock the execute_on_kaggle method
        self.mock_remote.execute_on_kaggle.return_value = {'success': True, 'output': 'Test output'}

        # Create a shell with mocks
        with patch('neural.cloud.interactive_shell.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = self.temp_dir.name
            self.shell = NeuralCloudShell('kaggle', self.mock_remote)
            self.shell.temp_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after the tests."""
        self.remote_patcher.stop()
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test shell initialization."""
        # Verify the remote connection was initialized
        self.assertEqual(self.shell.platform, 'kaggle')
        self.assertEqual(self.shell.remote, self.mock_remote)
        self.assertEqual(self.shell.kernel_id, 'test-kernel-id')

        # Verify connect_to_kaggle was called
        self.mock_remote.connect_to_kaggle.assert_called_once()

        # Verify create_kaggle_kernel was called
        self.mock_remote.create_kaggle_kernel.assert_called_once()

        # Verify execute_on_kaggle was called for initialization
        self.mock_remote.execute_on_kaggle.assert_called_once()

    def test_run_command(self):
        """Test the run command."""
        # Create a test file
        test_file = os.path.join(self.temp_dir.name, 'test.neural')
        with open(test_file, 'w') as f:
            f.write('network TestModel { input: (10, 10, 3) }')

        # Reset the mock
        self.mock_remote.execute_on_kaggle.reset_mock()

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_run(f'{test_file} --backend tensorflow --dataset MNIST --epochs 5')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_once()

        # Verify the arguments
        args, kwargs = self.mock_remote.execute_on_kaggle.call_args
        self.assertEqual(args[0], 'test-kernel-id')
        self.assertIn('network TestModel { input: (10, 10, 3) }', args[1])
        self.assertIn("backend='tensorflow'", args[1])
        self.assertIn("dataset='MNIST'", args[1])
        self.assertIn("epochs=5", args[1])

    def test_visualize_command(self):
        """Test the visualize command."""
        # Create a test file
        test_file = os.path.join(self.temp_dir.name, 'test.neural')
        with open(test_file, 'w') as f:
            f.write('network TestModel { input: (10, 10, 3) }')

        # Reset the mock
        self.mock_remote.execute_on_kaggle.reset_mock()

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_visualize(f'{test_file} --format png')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_once()

        # Verify the arguments
        args, kwargs = self.mock_remote.execute_on_kaggle.call_args
        self.assertEqual(args[0], 'test-kernel-id')
        self.assertIn('network TestModel { input: (10, 10, 3) }', args[1])
        self.assertIn("output_format='png'", args[1])

    def test_debug_command(self):
        """Test the debug command."""
        # Create a test file
        test_file = os.path.join(self.temp_dir.name, 'test.neural')
        with open(test_file, 'w') as f:
            f.write('network TestModel { input: (10, 10, 3) }')

        # Reset the mock
        self.mock_remote.execute_on_kaggle.reset_mock()

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_debug(f'{test_file} --backend tensorflow --no-tunnel')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_once()

        # Verify the arguments
        args, kwargs = self.mock_remote.execute_on_kaggle.call_args
        self.assertEqual(args[0], 'test-kernel-id')
        self.assertIn('network TestModel { input: (10, 10, 3) }', args[1])
        self.assertIn("backend='tensorflow'", args[1])
        self.assertIn("setup_tunnel=False", args[1])

    def test_shell_command(self):
        """Test the shell command."""
        # Reset the mock
        self.mock_remote.execute_on_kaggle.reset_mock()

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_shell('ls -la')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_once()

        # Verify the arguments
        args, kwargs = self.mock_remote.execute_on_kaggle.call_args
        self.assertEqual(args[0], 'test-kernel-id')
        self.assertIn('ls -la', args[1])

    def test_python_command(self):
        """Test the python command."""
        # Reset the mock
        self.mock_remote.execute_on_kaggle.reset_mock()

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_python('print("Hello, world!")')

        # Verify execute_on_kaggle was called
        self.mock_remote.execute_on_kaggle.assert_called_once()

        # Verify the arguments
        args, kwargs = self.mock_remote.execute_on_kaggle.call_args
        self.assertEqual(args[0], 'test-kernel-id')
        self.assertEqual(args[1], 'print("Hello, world!")')

    def test_history_command(self):
        """Test the history command."""
        # Add some history
        self.shell.history = [
            {'timestamp': 1234567890, 'command': 'run test.neural', 'success': True, 'output': 'Test output'},
            {'timestamp': 1234567891, 'command': 'visualize test.neural', 'success': False, 'output': 'Error'}
        ]

        # Run the command
        with patch('builtins.print') as mock_print:
            self.shell.do_history('')

        # Verify print was called
        mock_print.assert_called()

        # Verify the number of calls
        self.assertGreaterEqual(mock_print.call_count, 3)  # Header + 2 history items

    def test_exit_command(self):
        """Test the exit command."""
        # Run the command
        with patch('builtins.print') as mock_print:
            result = self.shell.do_exit('')

        # Verify the result
        self.assertTrue(result)

        # Verify delete_kaggle_kernel was called
        self.mock_remote.delete_kaggle_kernel.assert_called_once_with('test-kernel-id')

        # Verify cleanup was called
        self.mock_remote.cleanup.assert_called_once()

    def test_quit_command(self):
        """Test the quit command."""
        # Run the command
        with patch('builtins.print') as mock_print:
            result = self.shell.do_quit('')

        # Verify the result
        self.assertTrue(result)

        # Verify delete_kaggle_kernel was called
        self.mock_remote.delete_kaggle_kernel.assert_called_once_with('test-kernel-id')

        # Verify cleanup was called
        self.mock_remote.cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
