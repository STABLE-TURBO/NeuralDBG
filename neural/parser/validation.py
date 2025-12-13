"""
Parameter validation utilities for Neural DSL parser.
Provides strict type checking and conversion for layer parameters.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np

class ParamType(Enum):
    """Enumeration of parameter types supported in Neural DSL."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DICT = "dict"

T = TypeVar('T')

class ValidationError(Exception):
    """Exception raised for parameter validation errors."""
    pass

def validate_numeric(
    value: Any,
    param_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer_only: bool = False
) -> Union[int, float]:
    """
    Validate and convert a value to a numeric type.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        integer_only: Whether only integer values are allowed
        
    Returns:
        Validated and converted numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, (dict, list)):
            raise ValidationError(f"{param_name} must be a number, got {type(value).__name__}")
        
        # Try to convert to float first
        if isinstance(value, str):
            # Remove any whitespace
            value = value.strip()
        
        num_val = float(value)
        
        # Check if we need an integer
        if integer_only:
            if not float(num_val).is_integer():
                raise ValidationError(f"{param_name} must be an integer, got {num_val}")
            num_val = int(num_val)
            
        # Check bounds
        if min_value is not None and num_val < min_value:
            raise ValidationError(f"{param_name} must be >= {min_value}, got {num_val}")
        if max_value is not None and num_val > max_value:
            raise ValidationError(f"{param_name} must be <= {max_value}, got {num_val}")
            
        return int(num_val) if integer_only else num_val
        
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid {param_name}: {str(e)}")

def validate_units(
    value: Any,
    param_name: str = "units"
) -> int:
    """
    Validate and convert a units parameter value.
    Units must be positive integers.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated units value as integer
        
    Raises:
        ValidationError: If validation fails
    """
    result = validate_numeric(
        value,
        param_name,
        min_value=1,
        integer_only=True
    )
    return int(result)  # Safe cast since we specified integer_only=True

def validate_shape(
    value: Any,
    param_name: str = "shape"
) -> tuple[int, ...]:
    """
    Validate and convert a shape parameter value.
    Shape dimensions must be positive integers.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated shape as tuple of integers
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{param_name} must be a list or tuple, got {type(value).__name__}")
        
    try:
        dims = [int(validate_numeric(dim, f"{param_name} dimension", min_value=1, integer_only=True))
                for dim in value]
        return tuple(dims)
    except ValidationError as e:
        raise ValidationError(f"Invalid {param_name}: {str(e)}")

def validate_probability(
    value: Any,
    param_name: str = "probability"
) -> float:
    """
    Validate and convert a probability parameter value.
    Must be a float between 0 and 1.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated probability as float
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_numeric(
        value,
        param_name,
        min_value=0.0,
        max_value=1.0
    )

def validate_string(
    value: Any,
    param_name: str = "parameter",
    min_length: int = 0,
    max_length: int = 1000,
    allowed_pattern: Optional[str] = None
) -> str:
    """
    Validate a string parameter.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        min_length: Minimum string length
        max_length: Maximum string length
        allowed_pattern: Regex pattern the string must match
        
    Returns:
        Validated string
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param_name} must be a string, got {type(value).__name__}")
    
    if len(value) < min_length:
        raise ValidationError(f"{param_name} too short (min {min_length} chars), got {len(value)}")
    
    if len(value) > max_length:
        raise ValidationError(f"{param_name} too long (max {max_length} chars), got {len(value)}")
    
    if allowed_pattern:
        import re
        if not re.match(allowed_pattern, value):
            raise ValidationError(f"{param_name} does not match required pattern: {value}")
    
    return value

def validate_activation(
    value: Any,
    param_name: str = "activation"
) -> str:
    """
    Validate an activation function name.
    
    Args:
        value: Activation function name to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated activation function name
        
    Raises:
        ValidationError: If activation function is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param_name} must be a string, got {type(value).__name__}")
    
    value = value.lower().strip()
    
    valid_activations = {
        'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign',
        'elu', 'selu', 'swish', 'gelu', 'leaky_relu', 'prelu',
        'linear', 'none', 'hard_sigmoid', 'exponential'
    }
    
    if value not in valid_activations:
        raise ValidationError(
            f"Invalid {param_name} '{value}'. Supported: {', '.join(sorted(valid_activations))}"
        )
    
    return value

def validate_optimizer(
    value: Any,
    param_name: str = "optimizer"
) -> str:
    """
    Validate an optimizer name.
    
    Args:
        value: Optimizer name to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated optimizer name
        
    Raises:
        ValidationError: If optimizer is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param_name} must be a string, got {type(value).__name__}")
    
    value = value.lower().strip()
    
    valid_optimizers = {
        'sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta',
        'adamax', 'nadam', 'ftrl'
    }
    
    if value not in valid_optimizers:
        raise ValidationError(
            f"Invalid {param_name} '{value}'. Supported: {', '.join(sorted(valid_optimizers))}"
        )
    
    return value

def validate_layer_type(
    layer_type: Any
) -> str:
    """
    Validate a layer type name.
    
    Args:
        layer_type: Layer type to validate
        
    Returns:
        Validated layer type
        
    Raises:
        ValidationError: If layer type is invalid
    """
    if not isinstance(layer_type, str):
        raise ValidationError(f"Layer type must be a string, got {type(layer_type).__name__}")
    
    layer_type = layer_type.strip()
    
    if not layer_type:
        raise ValidationError("Layer type cannot be empty")
    
    if len(layer_type) > 100:
        raise ValidationError(f"Layer type too long (max 100 chars): {layer_type}")
    
    valid_layer_types = {
        'Dense', 'Conv1D', 'Conv2D', 'Conv3D', 'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
        'AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D', 'GlobalAveragePooling2D',
        'GlobalAveragePooling1D', 'Dropout', 'Flatten', 'LSTM', 'GRU', 'SimpleRNN',
        'BatchNormalization', 'LayerNormalization', 'Activation', 'Output', 'Embedding',
        'Attention', 'MultiHeadAttention', 'Transformer', 'TransformerEncoder', 'TransformerDecoder',
        'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Concatenate', 'Dot'
    }
    
    if layer_type not in valid_layer_types:
        raise ValidationError(
            f"Unknown layer type: {layer_type}. Supported types: {', '.join(sorted(valid_layer_types))}"
        )
    
    return layer_type

def validate_model_structure(model_data: Dict[str, Any]) -> None:
    """
    Validate the structure of a complete model specification.
    
    Args:
        model_data: Model data dictionary to validate
        
    Raises:
        ValidationError: If model structure is invalid
    """
    if not isinstance(model_data, dict):
        raise ValidationError(f"Model data must be a dictionary, got {type(model_data).__name__}")
    
    # Validate required fields
    required_fields = ['input', 'layers']
    for field in required_fields:
        if field not in model_data:
            raise ValidationError(f"Missing required field in model: {field}")
    
    # Validate input specification
    if not isinstance(model_data['input'], dict):
        raise ValidationError("Model input must be a dictionary")
    
    if 'shape' not in model_data['input']:
        raise ValidationError("Model input must specify a shape")
    
    try:
        validate_shape(model_data['input']['shape'], 'input shape')
    except ValidationError as e:
        raise ValidationError(f"Invalid input shape: {str(e)}")
    
    # Validate layers
    if not isinstance(model_data['layers'], list):
        raise ValidationError("Model layers must be a list")
    
    if len(model_data['layers']) == 0:
        raise ValidationError("Model must have at least one layer")
    
    if len(model_data['layers']) > 1000:
        raise ValidationError(f"Too many layers (max 1000), got: {len(model_data['layers'])}")
    
    # Validate each layer
    for i, layer in enumerate(model_data['layers']):
        try:
            validate_layer_config(layer, i)
        except ValidationError as e:
            raise ValidationError(f"Layer {i}: {str(e)}")
    
    # Validate optimizer if present
    if 'optimizer' in model_data and model_data['optimizer'] is not None:
        validate_optimizer_config(model_data['optimizer'])
    
    # Validate training config if present
    if 'training_config' in model_data and model_data['training_config'] is not None:
        validate_training_config(model_data['training_config'])

def validate_layer_config(layer: Dict[str, Any], layer_index: int = 0) -> None:
    """
    Validate a single layer configuration.
    
    Args:
        layer: Layer configuration dictionary
        layer_index: Index of layer in the model
        
    Raises:
        ValidationError: If layer configuration is invalid
    """
    if not isinstance(layer, dict):
        raise ValidationError(f"Layer must be a dictionary, got {type(layer).__name__}")
    
    if 'type' not in layer:
        raise ValidationError("Layer missing required 'type' field")
    
    # Validate layer type
    layer_type = validate_layer_type(layer['type'])
    
    # Validate parameters if present
    if 'params' in layer and layer['params'] is not None:
        if not isinstance(layer['params'], dict):
            raise ValidationError(f"Layer params must be a dictionary, got {type(layer['params']).__name__}")
        
        validate_layer_params(layer_type, layer['params'])

def validate_layer_params(layer_type: str, params: Dict[str, Any]) -> None:
    """
    Validate layer-specific parameters.
    
    Args:
        layer_type: Type of the layer
        params: Parameter dictionary to validate
        
    Raises:
        ValidationError: If parameters are invalid for the layer type
    """
    # Common validations for specific layer types
    if layer_type in ['Dense', 'Output']:
        if 'units' in params:
            validate_units(params['units'], 'units')
        
        if 'activation' in params and params['activation']:
            validate_activation(params['activation'], 'activation')
    
    elif layer_type in ['Conv1D', 'Conv2D', 'Conv3D']:
        if 'filters' in params:
            validate_numeric(params['filters'], 'filters', min_value=1, integer_only=True)
        
        if 'kernel_size' in params:
            validate_numeric(params['kernel_size'], 'kernel_size', min_value=1, integer_only=True)
        
        if 'strides' in params:
            validate_numeric(params['strides'], 'strides', min_value=1, integer_only=True)
    
    elif layer_type == 'Dropout':
        if 'rate' in params:
            validate_probability(params['rate'], 'dropout rate')
    
    elif layer_type in ['LSTM', 'GRU', 'SimpleRNN']:
        if 'units' in params:
            validate_units(params['units'], 'units')
        
        if 'return_sequences' in params:
            if not isinstance(params['return_sequences'], bool):
                raise ValidationError("return_sequences must be a boolean")

def validate_optimizer_config(optimizer: Dict[str, Any]) -> None:
    """
    Validate optimizer configuration.
    
    Args:
        optimizer: Optimizer configuration dictionary
        
    Raises:
        ValidationError: If optimizer configuration is invalid
    """
    if not isinstance(optimizer, dict):
        raise ValidationError(f"Optimizer must be a dictionary, got {type(optimizer).__name__}")
    
    if 'type' not in optimizer:
        raise ValidationError("Optimizer missing required 'type' field")
    
    validate_optimizer(optimizer['type'], 'optimizer type')
    
    if 'params' in optimizer and optimizer['params'] is not None:
        if not isinstance(optimizer['params'], dict):
            raise ValidationError("Optimizer params must be a dictionary")
        
        # Validate learning rate
        if 'learning_rate' in optimizer['params']:
            lr = optimizer['params']['learning_rate']
            if not isinstance(lr, dict):  # Not an HPO param
                validate_numeric(lr, 'learning_rate', min_value=1e-10, max_value=10.0)

def validate_training_config(config: Dict[str, Any]) -> None:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Raises:
        ValidationError: If training configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Training config must be a dictionary, got {type(config).__name__}")
    
    if 'epochs' in config:
        validate_numeric(config['epochs'], 'epochs', min_value=1, max_value=100000, integer_only=True)
    
    if 'batch_size' in config:
        if not isinstance(config['batch_size'], dict):  # Not an HPO param
            validate_numeric(config['batch_size'], 'batch_size', min_value=1, max_value=10000, integer_only=True)
    
    if 'validation_split' in config:
        validate_probability(config['validation_split'], 'validation_split')

def load_and_validate_config(file_path: str, max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """
    Load and validate a configuration file (JSON or YAML).
    
    Args:
        file_path: Path to configuration file
        max_size: Maximum file size in bytes (default 10MB)
        
    Returns:
        Parsed and validated configuration dictionary
        
    Raises:
        ValidationError: If file is invalid or configuration is malformed
    """
    if not isinstance(file_path, str) or not file_path:
        raise ValidationError("File path must be a non-empty string")
    
    # Check file existence
    if not os.path.exists(file_path):
        raise ValidationError(f"Configuration file not found: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValidationError(f"Configuration file too large: {file_size} bytes (max {max_size})")
    
    if file_size == 0:
        raise ValidationError("Configuration file is empty")
    
    # Determine file type
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, PermissionError) as e:
        raise ValidationError(f"Failed to read configuration file: {str(e)}")
    except UnicodeDecodeError:
        raise ValidationError("Configuration file must be UTF-8 encoded")
    
    # Parse based on file extension
    try:
        if ext == '.json':
            config = json.loads(content)
        elif ext in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ValidationError("YAML support not available. Install PyYAML: pip install pyyaml")
            config = yaml.safe_load(content)
        else:
            raise ValidationError(f"Unsupported configuration file format: {ext}. Use .json, .yaml, or .yml")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in configuration file: {str(e)}")
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in configuration file: {str(e)}")
    
    if not isinstance(config, dict):
        raise ValidationError("Configuration file must contain a JSON object or YAML mapping")
    
    # Validate configuration structure
    validate_config_schema(config)
    
    return config

def validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Validate the schema of a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValidationError: If configuration schema is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Configuration must be a dictionary, got {type(config).__name__}")
    
    # Check for suspicious or dangerous keys
    dangerous_keys = ['__import__', 'eval', 'exec', '__builtins__', '__globals__']
    for key in config.keys():
        if key in dangerous_keys or key.startswith('__'):
            raise ValidationError(f"Dangerous configuration key detected: {key}")
    
    # Validate model configuration if present
    if 'model' in config:
        if isinstance(config['model'], dict):
            validate_model_structure(config['model'])
    
    # Validate training configuration if present
    if 'training' in config:
        if not isinstance(config['training'], dict):
            raise ValidationError("Training configuration must be a dictionary")
        validate_training_config(config['training'])
    
    # Validate HPO configuration if present
    if 'hpo' in config:
        if not isinstance(config['hpo'], dict):
            raise ValidationError("HPO configuration must be a dictionary")
        validate_hpo_config(config['hpo'])
    
    # Validate deployment configuration if present
    if 'deployment' in config:
        if not isinstance(config['deployment'], dict):
            raise ValidationError("Deployment configuration must be a dictionary")
        validate_deployment_config(config['deployment'])

def validate_hpo_config(hpo_config: Dict[str, Any]) -> None:
    """
    Validate HPO configuration section.
    
    Args:
        hpo_config: HPO configuration dictionary
        
    Raises:
        ValidationError: If HPO configuration is invalid
    """
    if 'n_trials' in hpo_config:
        n_trials = hpo_config['n_trials']
        if not isinstance(n_trials, int) or n_trials < 1 or n_trials > 10000:
            raise ValidationError(f"n_trials must be an integer between 1 and 10000, got: {n_trials}")
    
    if 'timeout' in hpo_config:
        timeout = hpo_config['timeout']
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError(f"timeout must be a positive number, got: {timeout}")
    
    if 'backend' in hpo_config:
        backend = hpo_config['backend']
        if not isinstance(backend, str):
            raise ValidationError(f"backend must be a string, got: {type(backend).__name__}")
        
        valid_backends = {'pytorch', 'tensorflow', 'jax'}
        if backend.lower() not in valid_backends:
            raise ValidationError(f"Invalid backend '{backend}'. Supported: {', '.join(valid_backends)}")

def validate_deployment_config(deploy_config: Dict[str, Any]) -> None:
    """
    Validate deployment configuration section.
    
    Args:
        deploy_config: Deployment configuration dictionary
        
    Raises:
        ValidationError: If deployment configuration is invalid
    """
    if 'platform' in deploy_config:
        platform = deploy_config['platform']
        if not isinstance(platform, str):
            raise ValidationError(f"platform must be a string, got: {type(platform).__name__}")
        
        valid_platforms = {
            'local', 'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            'sagemaker', 'vertex_ai', 'azure_ml'
        }
        if platform.lower() not in valid_platforms:
            raise ValidationError(
                f"Invalid platform '{platform}'. Supported: {', '.join(sorted(valid_platforms))}"
            )
    
    if 'port' in deploy_config:
        port = deploy_config['port']
        if not isinstance(port, int) or port < 1024 or port > 65535:
            raise ValidationError(f"port must be between 1024 and 65535, got: {port}")
    
    if 'replicas' in deploy_config:
        replicas = deploy_config['replicas']
        if not isinstance(replicas, int) or replicas < 1 or replicas > 100:
            raise ValidationError(f"replicas must be between 1 and 100, got: {replicas}")
    
    if 'resources' in deploy_config:
        if not isinstance(deploy_config['resources'], dict):
            raise ValidationError("resources must be a dictionary")
        
        resources = deploy_config['resources']
        if 'cpu' in resources:
            cpu = resources['cpu']
            if not isinstance(cpu, (int, float, str)):
                raise ValidationError(f"cpu must be a number or string, got: {type(cpu).__name__}")
        
        if 'memory' in resources:
            memory = resources['memory']
            if not isinstance(memory, (int, str)):
                raise ValidationError(f"memory must be a number or string, got: {type(memory).__name__}")
        
        if 'gpu' in resources:
            gpu = resources['gpu']
            if not isinstance(gpu, (int, bool)):
                raise ValidationError(f"gpu must be an integer or boolean, got: {type(gpu).__name__}")