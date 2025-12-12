import logging
import re
from neural.parser.parser import ModelTransformer, create_parser
from typing import Any, Dict, Union, Optional
from neural.code_generation.tensorflow_generator import TensorFlowGenerator
from neural.code_generation.pytorch_generator import PyTorchGenerator
from neural.code_generation.onnx_generator import ONNXGenerator, export_onnx

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def to_number(x: str) -> Union[int, float]:
    try:
        return int(x)
    except ValueError:
        return float(x)


def generate_code(model_data: Dict[str, Any], backend: str, best_params: Optional[Dict[str, Any]] = None, auto_flatten_output: bool = False) -> str:
    """Generate code for the specified backend using the strategy pattern.
    
    Args:
        model_data: Dictionary containing model configuration
        backend: Backend framework ('tensorflow', 'pytorch', or 'onnx')
        best_params: Optional dictionary of optimized hyperparameters
        auto_flatten_output: Whether to automatically flatten higher-rank inputs
        
    Returns:
        Generated code as a string
        
    Raises:
        ValueError: If model_data format is invalid or backend is unsupported
    """
    if not isinstance(model_data, dict) or 'layers' not in model_data or 'input' not in model_data:
        raise ValueError("Invalid model_data format: must be a dict with 'layers' and 'input' keys")

    if backend == "tensorflow":
        generator = TensorFlowGenerator(model_data, best_params, auto_flatten_output)
        return generator.generate()
    elif backend == "pytorch":
        generator = PyTorchGenerator(model_data, best_params, auto_flatten_output)
        return generator.generate()
    elif backend == "onnx":
        return export_onnx(model_data, "model.onnx")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'tensorflow', 'pytorch', or 'onnx'.")


def save_file(filename: str, content: str) -> None:
    """Save content to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error writing file: {filename}. {e}")
    logger.info("Successfully saved file: %s", filename)


def load_file(filename: str) -> Any:
    """Load and parse a neural config file."""
    with open(filename, 'r') as f:
        content = f.read()
    if filename.endswith('.neural') or filename.endswith('.nr'):
        return create_parser('network').parse(content)
    elif filename.endswith('.rnr'):
        return create_parser('research').parse(content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def generate_optimized_dsl(config: str, best_params: Dict[str, Any]) -> str:
    """Generate optimized DSL code with the best hyperparameters."""
    try:
        transformer = ModelTransformer()
        _, hpo_params = transformer.parse_network_with_hpo(config)
        lines = config.strip().split('\n')

        logger.info(f"Initial lines: {lines}")
        logger.info(f"best_params: {best_params}")
        logger.info(f"hpo_params: {hpo_params}")

        for hpo in hpo_params:
            if hpo['layer_type'].lower() == 'training_config' and hpo['param_name'] == 'batch_size':
                param_key = 'batch_size'
            elif hpo['layer_type'].lower() == 'optimizer' and hpo['param_name'] == 'params.learning_rate':
                param_key = 'learning_rate'
            else:
                param_key = f"{hpo['layer_type'].lower()}_{hpo['param_name']}"

            if param_key not in best_params:
                logger.warning(f"Parameter {param_key} not found in best_params, skipping")
                continue

            if 'hpo' not in hpo or not hpo['hpo']:
                logger.warning(f"Missing 'hpo' data for parameter {param_key}, skipping")
                continue

            hpo_type = hpo['hpo'].get('type')
            if not hpo_type:
                logger.warning(f"Missing 'type' in hpo data for parameter {param_key}, skipping")
                continue

            if hpo_type in ('choice', 'categorical'):
                values = hpo['hpo'].get('original_values', hpo['hpo'].get('values', []))
                if not values:
                    logger.warning(f"Missing 'values' for choice/categorical parameter {param_key}, skipping")
                    continue
                hpo_str = f"choice({', '.join(map(str, values))})"
            elif hpo_type == 'range':
                start = hpo['hpo'].get('start')
                end = hpo['hpo'].get('end')
                original_parts = hpo['hpo'].get('original_parts', [])
                if not original_parts and (start is None or end is None):
                    logger.warning(f"Missing range bounds for parameter {param_key}, skipping")
                    continue
                if not original_parts:
                    original_parts = [str(start), str(end)]
                if 'step' in hpo['hpo']:
                    hpo_str = f"range({', '.join(original_parts)}, step={hpo['hpo']['step']})"
                else:
                    hpo_str = f"range({', '.join(original_parts)})"
            elif hpo_type == 'log_range':
                low = hpo['hpo'].get('original_low', str(hpo['hpo'].get('start', hpo['hpo'].get('min', ''))))
                high = hpo['hpo'].get('original_high', str(hpo['hpo'].get('end', hpo['hpo'].get('max', ''))))
                if not low or not high:
                    logger.warning(f"Missing log_range bounds for parameter {param_key}, skipping")
                    continue
                hpo_str = f"log_range({low}, {high})"
            else:
                logger.warning(f"Unknown HPO type: {hpo_type}, skipping")
                continue

            logger.info(f"Processing hpo: {hpo}, param_key: {param_key}, hpo_str: {hpo_str}")
            for i, line in enumerate(lines):
                full_hpo = f"HPO({hpo_str})"
                if full_hpo in line:
                    old_line = lines[i]
                    param_value = best_params[param_key]
                    if isinstance(param_value, (int, float)):
                        param_value_str = str(param_value)
                    elif isinstance(param_value, str):
                        param_value_str = f'"{param_value}"'
                    elif isinstance(param_value, dict):
                        if 'value' in param_value:
                            param_value_str = str(param_value['value'])
                        else:
                            logger.warning(f"Dictionary parameter without 'value' key: {param_value}, using string representation")
                            param_value_str = str(param_value)
                    else:
                        param_value_str = str(param_value)

                    new_line = line.replace(full_hpo, param_value_str)
                    lines[i] = new_line
                    logger.info(f"Replaced line {i}: '{old_line}' -> '{new_line}'")
                    break

        if 'learning_rate' in best_params:
            for i, line in enumerate(lines):
                if 'optimizer:' in line and 'learning_rate=HPO(' in line:
                    old_line = lines[i]
                    optimizer_type = re.search(r'optimizer:\s*(\w+)\(', old_line)
                    if optimizer_type:
                        opt_type = optimizer_type.group(1)
                        lr_value = best_params['learning_rate']
                        if isinstance(lr_value, (int, float)):
                            lr_str = str(lr_value)
                        elif isinstance(lr_value, str):
                            lr_str = f'"{lr_value}"'
                        elif isinstance(lr_value, dict):
                            if 'value' in lr_value:
                                lr_str = str(lr_value['value'])
                            else:
                                logger.warning(f"Dictionary parameter without 'value' key: {lr_value}, using string representation")
                                lr_str = str(lr_value)
                        else:
                            lr_str = str(lr_value)

                        new_line = f"        optimizer: {opt_type}(learning_rate={lr_str})"
                        lines[i] = new_line
                        logger.info(f"Replaced optimizer line {i}: '{old_line}' -> '{new_line}'")
                        break

        logger.info(f"Final lines: {lines}")
        return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Error generating optimized DSL: {str(e)}")
        raise
