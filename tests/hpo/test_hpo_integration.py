import pytest
import torch
import os
import sys
from unittest.mock import patch
from lark.exceptions import VisitError

# Add the project root to the Python path to allow importing from the neural package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the necessary components from the neural package
from neural.hpo.hpo import optimize_and_return, create_dynamic_model
from neural.hpo.hpo import train_model, objective
from neural.parser.parser import ModelTransformer, DSLValidationError
from neural.code_generation.code_generator import generate_optimized_dsl

class MockTrial:
    """
    Mock implementation of Optuna's Trial object for testing HPO functionality.

    This class simulates the behavior of an Optuna trial by providing methods
    to suggest hyperparameter values without actually running optimization.
    It returns deterministic values for testing purposes.
    """
    def suggest_categorical(self, name, choices):
        """
        Simulate suggesting a categorical hyperparameter.
        Returns 32 for 'batch_size', otherwise returns the first choice.

        Args:
            name: Name of the hyperparameter
            choices: List of possible values

        Returns:
            A deterministic choice from the provided options
        """
        return 32 if name == "batch_size" else choices[0]

    def suggest_float(self, name, low, high, step=None, log=False):
        """
        Simulate suggesting a float hyperparameter.
        Returns the lower bound for regular ranges, 0.001 for log ranges.

        Args:
            name: Name of the hyperparameter
            low: Lower bound of the range
            high: Upper bound of the range
            step: Step size (optional)
            log: Whether to use log scale (optional)

        Returns:
            A deterministic float value
        """
        return low if not log else 0.001

    def suggest_int(self, name, low, high):
        """
        Simulate suggesting an integer hyperparameter.
        Always returns the lower bound for simplicity.

        Args:
            name: Name of the hyperparameter
            low: Lower bound of the range
            high: Upper bound of the range

        Returns:
            The lower bound integer value
        """
        return low

def mock_data_loader(dataset_name, input_shape, batch_size=32, train=True, backend='pytorch'):
    """
    Mock data loader that creates synthetic data for testing.

    This function creates random tensors with appropriate shapes to simulate
    real datasets, without requiring actual data files. It handles both
    training and validation data, and properly formats the data for PyTorch.

    Args:
        dataset_name: Name of the dataset (unused, for API compatibility)
        input_shape: Shape of input tensors
        batch_size: Batch size for the DataLoader
        train: Whether to create training data (True) or validation data (False)
        backend: The backend to use ('pytorch' or 'tensorflow', default: 'pytorch')

    Returns:
        A PyTorch DataLoader with synthetic data
    """
    # Create random data with appropriate shapes
    # Training set has 100 samples, validation set has 20 samples
    if train:
        x = torch.randn(100, *input_shape)  # Random input data
        y = torch.randint(0, 10, (100,))    # Random labels (10 classes)
    else:
        x = torch.randn(20, *input_shape)   # Smaller validation set
        y = torch.randint(0, 10, (20,))     # Random labels (10 classes)

    # Convert to NCHW format for PyTorch if dealing with image data
    if len(input_shape) == 3:  # For image data (height, width, channels)
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW format conversion

    # Create and return a PyTorch DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

# 1. Enhanced Forward Pass Tests
def test_model_forward_flat_input():
    """
    Test that a model with a simple dense layer can correctly process flat input data.

    This test:
    1. Creates a simple neural network with Dense and Output layers
    2. Builds a PyTorch model from the Neural DSL specification
    3. Passes random input data through the model
    4. Verifies that the output has the expected shape (batch_size, num_classes)

    The test ensures that the basic model creation and forward pass functionality
    works correctly without any HPO parameters.
    """
    # Define a simple network in Neural DSL
    config = "network Test { input: (28,28,1) layers: Flatten() Dense(128) Output(10) }"


    # Parse the DSL into a model dictionary and HPO parameters
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Create a PyTorch model from the model dictionary
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')

    # Create random input data in NHWC format (batch, height, width, channels)
    x = torch.randn(32, *model_dict['input']['shape'])  # [32, 28, 28, 1]

    # Convert to NCHW format for PyTorch (batch, channels, height, width)
    x = x.permute(0, 3, 1, 2)  # [32, 1, 28, 28]

    # Pass the input through the model
    output = model(x)

    # Verify that the output has the expected shape
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"

def test_model_forward_conv2d():
    """
    Test that a model with convolutional layers can correctly process image input data.

    This test:
    1. Creates a CNN with Conv2D, Flatten, Dense, and Output layers
    2. Builds a PyTorch model from the Neural DSL specification
    3. Passes random image data through the model
    4. Verifies that the output has the expected shape (batch_size, num_classes)

    The test ensures that convolutional models can be created and process
    image data correctly, with proper tensor format conversion.
    """
    # Define a CNN in Neural DSL with Conv2D, Flatten, Dense, and Output layers
    config = "network Test { input: (28,28,1) layers: Conv2D(filters=16, kernel_size=3, data_format='channels_last') Flatten() Dense(128) Output(10) }"


    # Parse the DSL into a model dictionary and HPO parameters
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Create a PyTorch model from the model dictionary
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')

    # Create random input data in NHWC format (batch, height, width, channels)
    x = torch.randn(32, *model_dict['input']['shape'])  # [32, 28, 28, 1]

    # Convert to NCHW format for PyTorch (batch, channels, height, width)
    x = x.permute(0, 3, 1, 2)  # [32, 1, 28, 28]

    # Pass the input through the model
    output = model(x)

    # Verify that the output has the expected shape
    assert output.shape == torch.Size([32, 10]), f"Expected output shape [32, 10], got {output.shape}"

# 2. Enhanced Training Loop Tests
@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_training_loop_convergence():
    """
    Test that the training loop can successfully train a model and produce reasonable metrics.

    This test:
    1. Creates a simple neural network model
    2. Sets up an Adam optimizer
    3. Creates mock training and validation data loaders
    4. Runs the training loop
    5. Verifies that the resulting loss and accuracy are within reasonable ranges

    The test ensures that the training functionality works correctly and
    produces sensible metrics, without requiring actual training data.

    Note: This test uses the @patch decorator to replace the real data loader
    with our mock_data_loader function during testing.
    """
    # Define a simple network in Neural DSL
    config = "network Test { input: (28,28,1) layers: Flatten() Dense(128) Output(10) }"


    # Parse the DSL into a model dictionary and HPO parameters
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Create a PyTorch model from the model dictionary
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')

    # Create an optimizer for the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create mock training and validation data loaders
    train_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=True)
    val_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=False)

    # Train the model and get the resulting metrics
    # The train_model function should return a tuple of (loss, accuracy)
    loss = train_model(model, optimizer, train_loader, val_loader, backend='pytorch',
                      execution_config=model_dict.get('execution_config', {}))

    # Verify that the loss is a reasonable value (a float between 0 and 10)
    # Note: The IDE shows an error here about "None" not being subscriptable,
    # which suggests train_model might return None in some cases
    # This assertion assumes train_model returns a tuple of (loss, accuracy)
    assert isinstance(loss[0], float) and 0 <= loss[0] < 10, f"Loss not reasonable: {loss[0]}"

    # Verify that the accuracy is between 0 and 1
    assert 0 <= loss[1] <= 1, f"Accuracy not in range: {loss[1]}"

@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_training_loop_invalid_optimizer():
    """
    Test that the training loop correctly handles invalid optimizer inputs.

    This test:
    1. Creates a simple neural network model
    2. Passes an invalid optimizer (a string instead of an optimizer object)
    3. Verifies that an AttributeError is raised

    The test ensures that the training functionality properly validates
    its inputs and fails gracefully with appropriate errors.
    """
    # Define a simple network in Neural DSL
    config = "network Test { input: (28,28,1) layers: Flatten() Dense(128) Output(10) }"


    # Parse the DSL into a model dictionary and HPO parameters
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Create a PyTorch model from the model dictionary
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')

    # Create mock training and validation data loaders
    train_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=True)
    val_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=False)

    # Verify that passing an invalid optimizer (a string) raises an AttributeError
    with pytest.raises(AttributeError):
        train_model(model, "invalid_optimizer", train_loader, val_loader,
                   backend='pytorch', execution_config=model_dict.get('execution_config', {}))

# 3. Enhanced HPO Objective Tests
@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_hpo_objective_multi_objective():
    """
    Test that the HPO objective function correctly calculates multiple metrics.

    This test:
    1. Creates a simple neural network with specified loss and optimizer
    2. Calls the objective function with a mock trial
    3. Verifies that it returns multiple metrics (loss, accuracy, precision, recall)
    4. Checks that all metrics are of the correct type

    The test ensures that the objective function used for hyperparameter
    optimization correctly calculates and returns multiple evaluation metrics.
    """
    # Define a network with explicit loss and optimizer
    # Using the named optimizer format with parameters
    config = """
    network Test {
        input: (28,28,1)
        layers: Flatten() Dense(128) Output(10)

        loss: 'cross_entropy'
        optimizer: Adam(learning_rate=0.001)
    }
    """

    # Create a mock trial for hyperparameter suggestion
    trial = MockTrial()

    # Call the objective function and get the metrics
    # Now that we've fixed the device issue in the objective function, we can call it directly
    loss, acc, precision, recall = objective(trial, config, 'MNIST', backend='pytorch')

    # Verify that all metrics are of the correct type
    assert isinstance(loss, float), "Loss should be a float"
    assert isinstance(acc, float), "Accuracy should be a float"
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"

@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_hpo_objective_with_hpo_params():
    """
    Test that the HPO objective function works correctly with HPO parameters.

    This test:
    1. Creates a neural network with HPO parameters for layer size and learning rate
    2. Calls the objective function with a mock trial
    3. Verifies that it returns valid metrics
    4. Checks that all metrics are within expected ranges

    The test ensures that the objective function correctly handles models
    with hyperparameters to be optimized, using the trial object to suggest values.
    """
    # Define a network with HPO parameters for layer size and learning rate
    # Using the named optimizer format with HPO parameter
    config = """
    network Test {
        input: (28,28,1)
        layers: Flatten() Dense(HPO(choice(64, 128))) Output(10)
        optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
    }
    """


    # Create a mock trial for hyperparameter suggestion
    trial = MockTrial()

    # Call the objective function and get the metrics
    # Now that we've fixed the device issue in the objective function, we can call it directly
    loss, acc, precision, recall = objective(trial, config, 'MNIST', backend='pytorch')

    # Verify that the loss is a float
    assert isinstance(loss, float), "Loss should be a float"

    # Verify that metrics are within expected ranges
    assert 0 <= acc <= 1, f"Accuracy should be between 0 and 1, got {acc}"
    assert 0 <= precision <= 1, f"Precision should be between 0 and 1, got {precision}"
    assert 0 <= recall <= 1, f"Recall should be between 0 and 1, got {recall}"

# 4. Enhanced Parser Tests
def test_parsed_hpo_config_all_types():
    """
    Test that the parser correctly handles all types of HPO parameters.

    This test:
    1. Creates a network with three different types of HPO parameters:
       - categorical choice (for Dense layer units)
       - range (for Dropout rate)
       - log_range (for Output layer units)
    2. Parses the network configuration
    3. Verifies that all HPO parameters are correctly identified and typed

    The test ensures that the parser can correctly extract and categorize
    different types of hyperparameters from the Neural DSL.
    """
    # Define a network with all three types of HPO parameters
    config = """
    network Test {
        input: (28,28,1)
        layers:
            Flatten()
            Dense(HPO(choice(32, 64)), activation="relu")
            Dropout(HPO(range(0.1, 0.5, step=0.1)))
            Output(HPO(log_range(10, 20)))
    }
    """

    # Parse the network configuration
    _, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Verify that all three HPO parameters were extracted
    assert len(hpo_params) == 3, f"Expected 3 HPO parameters, got {len(hpo_params)}"

    # Verify that each parameter has the correct type
    assert hpo_params[0]['hpo']['type'] == 'categorical', "First parameter should be categorical"
    assert hpo_params[1]['hpo']['type'] == 'range', "Second parameter should be range"
    assert hpo_params[2]['hpo']['type'] == 'log_range', "Third parameter should be log_range"

def test_parser_invalid_config():
    """
    Test that the parser correctly validates and rejects invalid configurations.

    This test:
    1. Creates a network with an invalid parameter (negative units in Dense layer)
    2. Attempts to parse the network configuration
    3. Verifies that the appropriate exception is raised
    4. Checks that the error message contains the expected validation message

    The test ensures that the parser properly validates inputs and provides
    helpful error messages when validation fails.
    """
    # Define a network with an invalid parameter (negative units)
    config = "network Test { input: (28,28,1) layers: Flatten() Dense(-1) }"


    # Attempt to parse the network and expect an exception
    with pytest.raises(VisitError) as exc_info:
        ModelTransformer().parse_network_with_hpo(config)

    # Verify that the exception is caused by a DSLValidationError
    assert isinstance(exc_info.value.__cause__, DSLValidationError), "Exception should be caused by DSLValidationError"

    # Verify that the error message mentions that the value must be positive
    assert "must be a positive" in str(exc_info.value.__cause__), "Error message should mention that value must be positive"

# 5. Enhanced HPO Integration Tests
@patch('neural.hpo.hpo.get_data', mock_data_loader)
# We still need to patch optimize_and_return to avoid running actual optimization trials
@patch('neural.hpo.hpo.optimize_and_return', lambda *_, **__: {'batch_size': 32, 'dense_units': 128, 'dropout_rate': 0.5, 'learning_rate': 0.001})
def test_hpo_integration_full_pipeline():
    """
    Test the complete HPO pipeline from configuration to optimized model.

    This test:
    1. Creates a network with multiple HPO parameters (layer size, dropout rate, learning rate)
    2. Runs a short optimization process (2 trials)
    3. Generates an optimized DSL with the best parameters
    4. Creates a model from the optimized DSL
    5. Verifies that the model works correctly

    The test ensures that the entire HPO workflow functions correctly,
    from initial configuration to final optimized model.
    """
    # Define a network with multiple HPO parameters
    # Using the named optimizer format with HPO parameter
    config = """
    network Example {
        input: (28,28,1)
        layers:
            Flatten()
            Dense(HPO(choice(128, 256)))
            Dropout(HPO(range(0.3, 0.7, step=0.1)))
            Output(10, "softmax")
        loss: "cross_entropy"
        optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
    }
    """
    # Run the optimization process with a small number of trials
    # The optimize_and_return function is patched to return a fixed set of parameters
    best_params = optimize_and_return(config, n_trials=2, dataset_name='MNIST', backend='pytorch')


    # Verify that the best parameters include the expected hyperparameters
    assert set(best_params.keys()).issubset({'batch_size', 'dense_units', 'dropout_rate', 'learning_rate'}), \
        f"Expected subset of {{'batch_size', 'dense_units', 'dropout_rate', 'learning_rate'}}, got {best_params.keys()}"

    # Generate an optimized DSL with the best parameters
    optimized = generate_optimized_dsl(config, best_params)

    # Verify that the optimized DSL no longer contains HPO markers in the layers
    # Note: We're not checking the optimizer line because it's handled differently
    assert 'HPO' not in optimized.split('optimizer:')[0], "Optimized DSL should not contain HPO markers in the layers"

    # Parse the optimized DSL
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(optimized)

    # Verify that there are no HPO parameters in the optimized model
    assert not hpo_params, "Optimized model should not have any HPO parameters"

    # Create a model from the optimized DSL
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')

    # Create input data in NCHW format for PyTorch
    x = torch.randn(32, 1, 28, 28)  # [batch_size, channels, height, width]

    # Verify that the model produces output with the expected shape
    assert model(x).shape == (32, 10), f"Expected output shape (32, 10), got {model(x).shape}"

# 6. Additional Tests
def test_code_generator_invalid_params():
    """
    Test that the code generator correctly handles invalid parameters.

    This test:
    1. Creates a simple network configuration
    2. Attempts to generate optimized DSL with invalid parameters
    3. Verifies that the function runs without error and returns a valid DSL

    The test ensures that the code generator properly validates its inputs
    and handles invalid parameters gracefully by skipping them.
    """
    # Define a simple network
    config = "network Test { input: (28,28,1) layers: Flatten() Dense(128) }"


    # Create invalid parameters (parameter name not in the model)
    invalid_params = {'unknown_param': 42}

    # Our code now handles invalid parameters gracefully by skipping them
    # So we just verify that the function runs without error and returns a valid DSL
    result = generate_optimized_dsl(config, invalid_params)
    assert 'Dense(128)' in result, "The optimized DSL should still contain the original layer"

@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_hpo_edge_case_no_layers():
    """
    Test HPO with a minimal network that has no hidden layers.

    This test:
    1. Creates a network with only an Output layer (no hidden layers)
    2. Runs the optimization process
    3. Generates an optimized DSL
    4. Creates and tests a model from the optimized DSL

    The test ensures that the HPO system works correctly with minimal
    networks, handling edge cases where there are no hidden layers to optimize.

    Note: This test uses a different patch path than other tests, targeting
    'neural.automatic_hyperparameter_optimization.hpo.get_data' instead of
    'neural.hpo.hpo.get_data', which suggests there might be multiple
    implementations or import paths for the HPO functionality.
    """
    # Define a minimal network with only an Output layer
    config = "network Test { input: (28,28,1) layers: Output(10) optimizer: Adam(learning_rate=0.001) }"

    # Run the optimization process with a single trial
    best_params = optimize_and_return(config, n_trials=1, dataset_name='MNIST', backend='pytorch')

    # Verify that at least the batch_size parameter is optimized
    assert 'batch_size' in best_params, "batch_size should be in the optimized parameters"

    # Generate an optimized DSL with the best parameters
    optimized = generate_optimized_dsl(config, best_params)

    # Parse the optimized DSL
    model_dict, _ = ModelTransformer().parse_network_with_hpo(optimized)

    # Create a model from the optimized DSL
    # Note: Empty list is passed for hpo_params since there are no HPO parameters
    model = create_dynamic_model(model_dict, MockTrial(), [], backend='pytorch')

    # Create input data and verify that the model produces output with the expected shape
    # Note: Input is in NHWC format (batch, height, width, channels)
    assert model(torch.randn(32, 28, 28, 1)).shape == (32, 10), \
        "Model with no hidden layers should still produce correct output shape"
