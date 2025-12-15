import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from neural.exceptions import (
    DependencyError,
    InvalidParameterError,
    InvalidShapeError,
    ShapeException,
    ShapeMismatchError,
)
from neural.parser.parser import ModelTransformer

logger = logging.getLogger(__name__)

from .layer_docs import format_layer_documentation, get_layer_documentation
from .layer_handlers import (
    handle_add,
    handle_batch_normalization,
    handle_concatenate,
    handle_conv1d,
    handle_conv3d,
    handle_cropping2d,
    handle_dropout,
    handle_global_average_pooling1d,
    handle_lstm,
    handle_permute,
    handle_positional_encoding,
    handle_reshape,
    handle_zero_padding2d,
)
from .utils import (
    calculate_memory_usage,
    calculate_output_dims,
    detect_shape_issues,
    extract_param,
    format_error_message,
    format_memory_size,
    suggest_optimizations,
)


# Lazy load heavy dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    Digraph = None
    GRAPHVIZ_AVAILABLE = False

# Make torch optional - allows tests to run without torch installed
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

class PerformanceMonitor:
    def __init__(self):
        self.resource_history = []

    def monitor_resources(self):
        """Monitor CPU, memory, and GPU usage."""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_memory = 0
        # Check if torch is available and CUDA is available before accessing GPU memory
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        io_counters = psutil.disk_io_counters()
        io_usage = (io_counters.read_bytes + io_counters.write_bytes) / (1024 ** 2)  # MB

        self.resource_history.append({
            "timestamp": time.time(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_memory": gpu_memory,
            "io_usage": io_usage
        })
        return self.resource_history[-1]


class ShapePropagator:
    # Registry for external layer handlers
    LAYER_HANDLERS = {}

    @classmethod
    def register_layer_handler(cls, layer_type):
        """Decorator to register layer handlers dynamically.

        Args:
            layer_type: Type of layer to register handler for

        Returns:
            Decorator function
        """
        def decorator(func):
            cls.LAYER_HANDLERS[layer_type] = func
            return func
        return decorator

    def __init__(self, debug=False):
        self.debug = debug
        self.shape_history = []
        self.layer_connections = []
        self.current_layer = 0
        self.execution_trace = []  # Stores nntrace logs
        self.performance_monitor = PerformanceMonitor()
        self.hub = None
        self.issues = []  # Store detected issues
        self.optimizations = []  # Store optimization suggestions

        # Framework compatibility mappings
        self.param_aliases = {
            'Conv2D': {'filters': 'out_channels', 'kernel_size': 'kernel_size'},
            'BatchNormalization': {'axis': 'dim'},
            'Dense': {'units': 'out_features'},
            'LSTM': {'units': 'hidden_size'},
            'GRU': {'units': 'hidden_size'},
            'BatchNormalization': {'momentum': 'decay'}
        }

        # Initialize visualization (lazy)
        self.dot = None

    def propagate(self, input_shape: Tuple[Optional[int], ...],
              layer: Dict[str, Any],
              framework: str = 'tensorflow') -> Tuple[Optional[int], ...]:
        """Processes a layer and logs shape changes for nntrace."""
        # Validate layer has a type
        if "type" not in layer:
            # Handle malformed layer structure (e.g., from parser issues)
            if len(layer) == 1:
                # Try to extract type from the malformed structure
                key = next(iter(layer.keys()))
                if hasattr(key, 'value'):
                    layer_type = key.value
                    params = layer[key][0] if layer[key] else {}
                else:
                    raise InvalidParameterError(
                        parameter='type',
                        value=None,
                        expected="Layer must have a 'type' field"
                    )
            else:
                raise InvalidParameterError(
                    parameter='type',
                    value=None,
                    expected="Layer must have a 'type' field"
                )
        else:
            layer_type = layer["type"]
            params = layer.get("params", {})

        # Debug logging
        logger.debug(f"ShapePropagator.propagate - input_shape: {input_shape}, layer_type: {layer_type}")
        logger.debug(f"ShapePropagator.propagate - params: {params}")

        # Validate input shape
        if not input_shape:
            raise InvalidShapeError(
                "Input shape cannot be empty",
                input_shape=input_shape,
                layer_type=layer_type
            )

        # Check for negative dimensions in input shape
        if any(dim is not None and dim < 0 for dim in input_shape):
            raise InvalidShapeError(
                f"Input shape cannot contain negative dimensions: {input_shape}",
                input_shape=input_shape,
                layer_type=layer_type
            )

        # Validate layer parameters based on layer type
        self._validate_layer_params(layer_type, params, input_shape, framework)

        # Only set kernel_size for layers that need it

        if layer_type in ['Conv2D', 'MaxPooling2D']:  # Add other layers as needed
            kernel_size = params.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            elif isinstance(kernel_size, list):
                kernel_size = tuple(kernel_size)
            elif isinstance(kernel_size, dict):
                logger.debug(f"ShapePropagator.propagate - kernel_size is a dict: {kernel_size}")
                # If it's a dictionary with a 'value' key, use that value
                if 'value' in kernel_size:
                    kernel_size = (kernel_size['value'], kernel_size['value'])
                # Otherwise, use a default value
                else:
                    logger.debug(f"ShapePropagator.propagate - kernel_size dict without 'value' key, using default")
                    kernel_size = (3, 3)  # Default value
            params["kernel_size"] = kernel_size  # Ensure tuple in params

        if layer_type == 'MultiHeadAttention':
            return input_shape
        
        prev_layer = self.current_layer if self.current_layer > 0 else None  # Track previous layer for connections

        # Check for registered external handlers first
        if layer_type in self.LAYER_HANDLERS:
            output_shape = self.LAYER_HANDLERS[layer_type](input_shape, params, self)
        # Standard shape propagation logic
        elif layer_type == "Conv2D":
            output_shape = self._handle_conv2d(input_shape, params)
        elif layer_type == "MaxPooling2D":
            output_shape = self._handle_maxpooling2d(input_shape, params)
        elif layer_type == "Dense":
            output_shape = self._handle_dense(input_shape, params)
        elif layer_type == "Output":
            output_shape = self._handle_output(input_shape, params)
        elif layer_type == "Embedding":
            output_shape = self._handle_embedding(input_shape, params)
        elif layer_type == "GlobalAveragePooling2D":
            output_shape = self._handle_globalaveragepooling2d(input_shape, params)
        elif layer_type == "UpSampling2D":
            output_shape = self._handle_upsampling2d(input_shape, params)
        elif layer_type == "MultiHeadAttention":
            output_shape = self._handle_multiheadattention(input_shape, params)
        # Use external handlers
        elif layer_type == "Conv1D":
            output_shape = handle_conv1d(input_shape, params)
        elif layer_type == "Conv3D":
            output_shape = handle_conv3d(input_shape, params)
        elif layer_type == "Dropout":
            output_shape = handle_dropout(input_shape, params)
        elif layer_type == "Flatten":
            output_shape = (input_shape[0], np.prod([d for d in input_shape[1:] if d is not None]))
        elif layer_type == "BatchNormalization":
            output_shape = handle_batch_normalization(input_shape, params, framework)
        elif layer_type == "Add":
            output_shape = handle_add(input_shape, params)
        elif layer_type == "Concatenate":
            output_shape = handle_concatenate(input_shape, params)
        elif layer_type == "Reshape":
            output_shape = handle_reshape(input_shape, params)
        elif layer_type == "Permute":
            output_shape = handle_permute(input_shape, params)
        elif layer_type == "ZeroPadding2D":
            output_shape = handle_zero_padding2d(input_shape, params)
        elif layer_type == "Cropping2D":
            output_shape = handle_cropping2d(input_shape, params)
        elif layer_type == "LSTM":
            output_shape = handle_lstm(input_shape, params)
        elif layer_type == "GlobalAveragePooling1D":
            output_shape = handle_global_average_pooling1d(input_shape, params)
        elif layer_type == "PositionalEncoding":
            output_shape = handle_positional_encoding(input_shape, params)
        elif layer_type == "TransformerEncoder":
            output_shape = input_shape
        elif layer_type == "TransformerDecoder":
            output_shape = input_shape
        else:
            output_shape = input_shape

        # Performance and resource tracking
        flops, memory, exec_time, comp_time, trans_time = self._compute_performance(layer, input_shape, output_shape)

        # Store shape history
        self.shape_history.append({"layer": layer_type, "input_shape": input_shape, "output_shape": output_shape})

        # nntrace integration: log execution trace with resource usage
        trace_entry = {
            "layer": layer_type,
            "execution_time": exec_time,
            "compute_time": comp_time,
            "transfer_time": trans_time,
            "kernel_size": params.get("kernel_size", (1, 1)) if "kernel_size" in params else (1, 1),
            "flops": flops,
            "memory": memory,
            "grad_norm": 0.0,  # Placeholder
            "dead_ratio": 0.0,  # Placeholder
            "mean_activation": 0.0,  # Placeholder
            "anomaly": False  # Placeholder
        }

        resources = self.performance_monitor.monitor_resources()
        trace_entry.update({
            "cpu_usage": resources["cpu_usage"],
            "memory_usage": resources["memory_usage"],
            "gpu_memory": resources["gpu_memory"],
            "io_usage": resources["io_usage"]
        })

        if self.debug:
            logger.debug(f"TRACE: {trace_entry}")

        self._visualize_layer(layer_type, output_shape)  # Creates node and increments self.current_layer
        if prev_layer is not None:
            self._create_connection(prev_layer, self.current_layer - 1)  # Connect previous to current
        return output_shape

###############################
### Performance Computation ###
###############################

    def _compute_performance(self, layer: dict, input_shape: tuple, output_shape: tuple) -> tuple:
        """Compute performance metrics (FLOPs, memory usage, etc.)."""
        # Extract layer type and params from layer dict
        layer_type = layer.get("type", "Unknown")
        params = layer.get("params", {})
        
        # Replace None with 1 to avoid NoneType math errors
        input_shape_calc = tuple(1 if dim is None else dim for dim in input_shape)
        output_shape_calc = tuple(1 if dim is None else dim for dim in output_shape)

        # Compute FLOPs (floating point operations)
        flops = 0
        if layer_type == "Conv2D":
            kernel_size = params.get("kernel_size", (3, 3))
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            filters = params.get("filters", 32)
            if isinstance(filters, dict) and 'value' in filters:
                filters = filters['value']
            h_out, w_out = output_shape[2:] if len(output_shape) > 3 else (1, 1)
            # Each output pixel requires kernel_size*kernel_size*in_channels operations
            in_channels = input_shape[1] if len(input_shape) > 1 else 1
            flops = h_out * w_out * kernel_size[0] * kernel_size[1] * in_channels * filters
        elif layer_type == "Dense":
            units = params.get("units", 64)
            if isinstance(units, dict) and 'value' in units:
                units = units['value']
            in_features = input_shape[-1] if input_shape else 1
            flops = in_features * units

        # Handle malformed layer structure for params
        if "params" not in layer:
            if len(layer) == 1:
                key = next(iter(layer.keys()))
                params = layer[key][0] if layer[key] else {}
            else:
                params = {}
        else:
            params = layer['params']

        # Initialize FLOPs
        flops = 0
        
        # FLOPs calculation based on layer type
        if layer_type == 'Conv2D':
            kernel_size = extract_param(params, 'kernel_size', (3, 3))
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            filters = extract_param(params, 'filters', 32)
            # FLOPs = 2 * kernel_h * kernel_w * input_channels * output_h * output_w * output_channels
            if len(input_shape_calc) >= 4 and len(output_shape_calc) >= 4:
                # Assuming channels_last format for calculation
                input_channels = input_shape_calc[-1] if len(input_shape_calc) == 4 else input_shape_calc[1]
                output_h, output_w = output_shape_calc[1:3] if len(output_shape_calc) == 4 else output_shape_calc[2:4]
                flops = 2 * kernel_size[0] * kernel_size[1] * input_channels * output_h * output_w * filters
        elif layer_type == 'Dense':
            units = extract_param(params, 'units', 64)
            # FLOPs = 2 * input_features * output_features
            if len(input_shape_calc) >= 2:
                input_features = input_shape_calc[-1]
                flops = 2 * input_features * units
        elif layer_type == 'LSTM' or layer_type == 'GRU':
            units = extract_param(params, 'units', 128)
            # FLOPs for LSTM = 4 * (input_size + hidden_size) * hidden_size * seq_len
            # FLOPs for GRU = 3 * (input_size + hidden_size) * hidden_size * seq_len
            if len(input_shape_calc) >= 3:
                batch, seq_len, input_size = input_shape_calc[0], input_shape_calc[1], input_shape_calc[2]
                gate_count = 4 if layer_type == 'LSTM' else 3
                flops = gate_count * (input_size + units) * units * seq_len
        elif layer_type in ['MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D']:
            # Pooling operations have minimal FLOPs
            flops = np.prod(output_shape_calc)
        elif layer_type in ['BatchNormalization', 'Dropout', 'Flatten']:
            # These operations have minimal computational cost
            flops = np.prod(output_shape_calc)

        # Memory usage (output tensor size in MB, assuming float32)
        memory_usage = np.prod(output_shape_calc) * 4 / (1024 ** 2)  # 4 bytes per float32

        # Simplified timing estimates
        compute_time = flops / 1e9  # Assuming 1 GFLOP/s
        transfer_time = memory_usage * 1e3 / 1e9  # Assuming 1 GB/s bandwidth
        exec_time = compute_time + transfer_time  # Total execution time

        return flops, memory_usage, exec_time, compute_time, transfer_time

##################################################
### Send execution trace data to the dashboard ###
##################################################
    def get_trace(self):
        """Retrieve the execution trace in a format compatible with nntrace."""
        trace = []
        for entry in self.execution_trace:
            if isinstance(entry, dict):
                # New format: entry is a dictionary
                layer_type = entry.get("layer", "Unknown")
                exec_time = entry.get("execution_time", 0.0)
                comp_time = entry.get("compute_time", 0.0)
                trans_time = entry.get("transfer_time", 0.0)
                flops = entry.get("flops", 0)
                memory = entry.get("memory", 0)
                grad_norm = entry.get("grad_norm", 0.0)
                dead_ratio = entry.get("dead_ratio", 0.0)
                mean_act = entry.get("mean_activation", 0.0)
                anomaly = entry.get("anomaly", False)
                params = entry.get("params", {})

                # Extract kernel_size from params or entry
                if "kernel_size" in entry:
                    kernel_size = entry.get("kernel_size", (1, 1))
                else:
                    kernel_size = params.get("kernel_size", (1, 1))
            else:
                # Old format: entry is a tuple
                try:
                    layer_type, exec_time, comp_time, trans_time, params, flops, memory, grad_norm, dead_ratio, mean_act, anomaly = entry
                except ValueError:
                    logger.warning(f"Invalid trace entry format: {entry}")
                    continue

                kernel_size = params.get("kernel_size", (1, 1)) if isinstance(params, dict) else (1, 1)

            # Ensure kernel_size is a tuple
            if isinstance(kernel_size, list):
                logger.warning(f"Converting list kernel_size {kernel_size} to tuple for {layer_type}")
                kernel_size = tuple(kernel_size)
            elif not isinstance(kernel_size, tuple):
                logger.warning(f"Unexpected kernel_size type {type(kernel_size)} for {layer_type}, defaulting to (1, 1)")
                kernel_size = (1, 1)

            trace.append({
                "layer": layer_type, "execution_time": exec_time, "compute_time": comp_time,
                "transfer_time": trans_time, "kernel_size": kernel_size,
                "flops": flops, "memory": memory, "grad_norm": grad_norm, "dead_ratio": dead_ratio,
                "mean_activation": mean_act, "anomaly": anomaly
            })
        return trace

#############################
### Visualization Methods ###
#############################

    def _visualize_layer(self, layer_type: str, output_shape: Tuple[Optional[int], ...]):
        """Create a node in the graph for the layer."""
        if not GRAPHVIZ_AVAILABLE:
            return
        
        if self.dot is None:
            self.dot = Digraph(comment='Neural Network Architecture')
            self.dot.attr('node', shape='record', style='filled', fillcolor='lightgrey')
        
        label = f"{layer_type}\\nOutput: {output_shape}"
        self.dot.node(str(self.current_layer), label=label)
        self.current_layer += 1

    def _create_connection(self, from_layer: int, to_layer: int):
        """Create an edge between two layers."""
        if GRAPHVIZ_AVAILABLE and self.dot is not None:
            self.dot.edge(str(from_layer), str(to_layer))
        self.layer_connections.append((from_layer, to_layer))

########################
### Getter Functions ###
########################

    def get_shape_history(self):
        return self.shape_history

    def get_layer_connections(self):
        return self.layer_connections

#########################
### Report Generation ###
#########################

    def generate_report(self):
        """Generate a comprehensive report including visualizations."""
        if not PLOTLY_AVAILABLE:
            raise DependencyError(
                dependency="plotly",
                feature="report generation",
                install_hint="pip install plotly"
            )
        
        # Create Plotly bar chart for parameter counts
        if not self.shape_history:
            layers = ["No data"]
            param_counts = [0]
        else:
            layers = [h["layer"] for h in self.shape_history]
            param_counts = [np.prod([d for d in h["output_shape"] if d is not None]) for h in self.shape_history]

        fig = go.Figure(data=[
            go.Bar(name='Parameters', x=layers, y=param_counts)
        ])
        fig.update_layout(title='Parameter Count by Layer', xaxis_title='Layer', yaxis_title='Parameter Count')

        # Detect and report issues
        self.issues = detect_shape_issues(self.shape_history)
        
        # Generate optimization suggestions
        self.optimizations = suggest_optimizations(self.shape_history)

        # Then check for internal handlers
        handler_name = f"_handle_{layer_type.lower()}"
        if hasattr(self, handler_name):
            output_shape = getattr(self, handler_name)(input_shape, params)
        else:
            # Try to use imported handlers
            if layer_type == 'Conv1D':
                output_shape = handle_conv1d(input_shape, params)
            elif layer_type == 'Conv3D':
                output_shape = handle_conv3d(input_shape, params)
            elif layer_type == 'LSTM':
                output_shape = handle_lstm(input_shape, params)
            elif layer_type == 'GRU':
                output_shape = self._handle_gru(input_shape, params)
            elif layer_type == 'Dropout':
                output_shape = handle_dropout(input_shape, params)
            elif layer_type == 'BatchNormalization':
                output_shape = handle_batch_normalization(input_shape, params)
            elif layer_type == 'Reshape':
                output_shape = handle_reshape(input_shape, params)
            elif layer_type == 'Permute':
                output_shape = handle_permute(input_shape, params)
            elif layer_type == 'ZeroPadding2D':
                output_shape = handle_zero_padding2d(input_shape, params)
            elif layer_type == 'Cropping2D':
                output_shape = handle_cropping2d(input_shape, params)
            elif layer_type == 'GlobalAveragePooling1D':
                output_shape = handle_global_average_pooling1d(input_shape, params)
            elif layer_type == 'MultiHeadAttention':
                output_shape = self._handle_multiheadattention(input_shape, params)
            elif layer_type == 'PositionalEncoding':
                output_shape = handle_positional_encoding(input_shape, params)
            else:
                # Fall back to default handler
                output_shape = self._handle_default(input_shape, params)

##################################
### Parameter Validation Logic ###
##################################

    def _validate_layer_params(self, layer_type: str, params: Dict[str, Any], input_shape: Tuple[Optional[int], ...], framework: str):
        """Validate layer parameters based on layer type and framework."""
        # Set default data_format based on framework
        if 'data_format' not in params:
            params['data_format'] = 'channels_first' if framework == 'pytorch' else 'channels_last'

        # Validate layer-specific parameters
        if layer_type == "Conv2D":
            # Ensure required params exist
            if 'filters' not in params:
                params['filters'] = 32  # Default
            if 'kernel_size' not in params:
                params['kernel_size'] = (3, 3)  # Default
            if 'stride' not in params:
                params['stride'] = 1  # Default
            if 'padding' not in params:
                params['padding'] = 'valid'  # Default

            # Validate filters
            filters = params['filters']
            if isinstance(filters, dict) and 'value' in filters:
                filters = filters['value']
            if not isinstance(filters, int) or filters <= 0:
                raise InvalidParameterError(
                    parameter='filters',
                    value=filters,
                    layer_type=layer_type,
                    expected="positive integer"
                )

        elif layer_type == "MaxPooling2D":
            if 'pool_size' not in params:
                params['pool_size'] = (2, 2)  # Default

        elif layer_type in ["Dense", "Output"]:
            if 'units' not in params:
                raise InvalidParameterError(
                    parameter='units',
                    value=None,
                    layer_type=layer_type,
                    expected="positive integer"
                )

####################################################################
### Shape propagation through 2 Dimensional Convolutional Layers ###
####################################################################

    def _handle_conv2d(self, input_shape, params):
        logger.debug(f"_handle_conv2d - input_shape: {input_shape}, params: {params}")
        data_format = params['data_format']  # 'channels_first' for PyTorch
        if data_format == 'channels_first':
            spatial_dims = input_shape[2:]  # Should be (28, 28)
        else:
            spatial_dims = input_shape[1:3]

        logger.debug(f"_handle_conv2d - spatial_dims: {spatial_dims}")

        kernel = params['kernel_size']
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif isinstance(kernel, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in kernel:
                kernel_value = kernel['value']
                if isinstance(kernel_value, int):
                    kernel = (kernel_value, kernel_value)
                else:
                    kernel = (3, 3)  # Default value
            # Otherwise, use default value
            else:
                logger.debug(f"_handle_conv2d - kernel is a dict without 'value' key: {kernel}, using default")
                kernel = (3, 3)  # Default value
        elif not isinstance(kernel, tuple):
            logger.debug(f"_handle_conv2d - Invalid kernel_size type: {type(kernel)}, value: {kernel}, using default")
            kernel = (3, 3)  # Default value

        stride = params.get('stride', 1)
        # Handle dictionary values in stride
        if isinstance(stride, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in stride:
                stride = stride['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_conv2d - stride is a dict without 'value' key: {stride}, using default")
                stride = 1  # Default value

        # Normalize stride to tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, (list, tuple)):
            stride = tuple(stride)
        else:
            stride = (1, 1)

        # Calculate padding with improved handling
        padding = self._calculate_padding(params, spatial_dims, kernel, stride)

        if isinstance(padding, int):
            padding = (padding, padding)
        elif isinstance(padding, (list, tuple)):
            padding = tuple(padding)
        elif isinstance(padding, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in padding:
                padding_value = padding['value']
                if isinstance(padding_value, int):
                    padding = (padding_value, padding_value)
                else:
                    padding = (0, 0)  # Default value
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_conv2d - padding is a dict without 'value' key: {padding}, using default")
                padding = (0, 0)  # Default value

        # Ensure padding has same length as spatial_dims
        if len(padding) == 1:
            padding = (padding[0], padding[0])

        logger.debug(f"_handle_conv2d - kernel: {kernel}, stride: {stride}, padding: {padding}")

        # Handle None dimensions
        output_spatial = []
        for i, (dim, k, s, pad) in enumerate(zip(spatial_dims, kernel, stride, padding)):
            if dim is None:
                output_spatial.append(None)
            else:
                out_dim = (dim + 2*pad - k) // s + 1
                if out_dim <= 0:
                    logger.debug(f"_handle_conv2d - Invalid output dimension at index {i}: {out_dim}, using 1")
                    out_dim = 1
                output_spatial.append(out_dim)

        filters = params['filters']
        # Handle dictionary values in filters
        if isinstance(filters, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in filters:
                filters = filters['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_conv2d - filters is a dict without 'value' key: {filters}, using default")
                filters = 32  # Default value

        logger.debug(f"_handle_conv2d - output_spatial: {output_spatial}, filters: {filters}")

        if data_format == 'channels_first':
            return (input_shape[0], filters, *output_spatial)
        else:
            return (input_shape[0], *output_spatial, filters)

    def _handle_maxpooling2d(self, input_shape, params):
        logger.debug(f"_handle_maxpooling2d - input_shape: {input_shape}, params: {params}")
        data_format = params.get('data_format', 'channels_last')
        pool_size = params['pool_size']

        # Handle dictionary values in pool_size
        if isinstance(pool_size, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in pool_size:
                pool_size_value = pool_size['value']
                if isinstance(pool_size_value, int):
                    pool_size = (pool_size_value, pool_size_value)
                else:
                    pool_size = (2, 2)  # Default value
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_maxpooling2d - pool_size is a dict without 'value' key: {pool_size}, using default")
                pool_size = (2, 2)  # Default value
        elif isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)

        # Get stride (defaults to pool_size if not specified)
        stride = params.get('stride', pool_size)
        # Handle dictionary values in stride
        if isinstance(stride, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in stride:
                stride_value = stride['value']
                if isinstance(stride_value, int):
                    stride_h = stride_w = stride_value
                else:
                    stride_h = stride_w = pool_size[0]  # Default value
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_maxpooling2d - stride is a dict without 'value' key: {stride}, using default")
                stride_h = stride_w = pool_size[0]  # Default value
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        elif isinstance(stride, (list, tuple)):
            stride_h, stride_w = stride[0], stride[1]

        logger.debug(f"_handle_maxpooling2d - pool_size: {pool_size}, stride_h: {stride_h}, stride_w: {stride_w}")

        # Calculate spatial dimensions based on data format
        if data_format == 'channels_last':
            # TensorFlow: input_shape = (batch, height, width, channels)
            if len(input_shape) >= 4:  # Ensure we have enough dimensions
                h, w = input_shape[1], input_shape[2]
                new_height = h // stride_h if h is not None else None
                new_width = w // stride_w if w is not None else None
                return (input_shape[0], new_height, new_width, input_shape[3])
            else:
                logger.debug(f"_handle_maxpooling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], 1, 1, input_shape[-1] if len(input_shape) > 1 else 1)
        else:
            # PyTorch: input_shape = (batch, channels, height, width)
            if len(input_shape) >= 4:  # Ensure we have enough dimensions
                h, w = input_shape[2], input_shape[3]
                new_height = h // stride_h if h is not None else None
                new_width = w // stride_w if w is not None else None
                return (input_shape[0], input_shape[1], new_height, new_width)
            else:
                logger.debug(f"_handle_maxpooling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], input_shape[1] if len(input_shape) > 1 else 1, 1, 1)

    def _handle_flatten(self, input_shape, params):
        # If there is a batch dimension, keep it.
        if len(input_shape) >= 1:
            batch = input_shape[0]
            # Multiply all dimensions after the batch dimension, handling None
            dims_to_flatten = [dim for dim in input_shape[1:] if dim is not None]
            if dims_to_flatten:
                flattened = np.prod(dims_to_flatten)
            else:
                flattened = 1
            return (batch, flattened)
        else:
            # Handle None values in single dimension
            prod_val = np.prod([dim for dim in input_shape if dim is not None])
            return (prod_val,)


    def _handle_dense(self, input_shape, params):
        logger.debug(f"_handle_dense - input_shape: {input_shape}, params: {params}")
        units = params['units']
        # Handle dictionary values in units
        if isinstance(units, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in units:
                units = units['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_dense - units is a dict without 'value' key: {units}, using default")
                units = 64  # Default value
        logger.debug(f"_handle_dense - units after processing: {units}")
        # Dense layer always outputs (batch, units)
        return (input_shape[0], units)

    def _handle_output(self, input_shape, params):
        logger.debug(f"_handle_output - input_shape: {input_shape}, params: {params}")
        units = params['units']
        # Handle dictionary values in units
        if isinstance(units, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in units:
                units = units['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_output - units is a dict without 'value' key: {units}, using default")
                units = 10  # Default value
        logger.debug(f"_handle_output - units after processing: {units}")
        # Output layer always outputs (batch, units)
        return (input_shape[0], units)

    def _handle_gru(self, input_shape, params):
        """Handle GRU layer shape propagation.
        
        Args:
            input_shape: Input tensor shape (batch, seq_len, input_size)
            params: Layer parameters
            
        Returns:
            Output tensor shape
        """
        logger.debug(f"_handle_gru - input_shape: {input_shape}, params: {params}")
        
        units = extract_param(params, 'units', 128)
        return_sequences = extract_param(params, 'return_sequences', False)

        if len(input_shape) < 3:
            return input_shape  # Invalid input shape, return unchanged

        batch_size = input_shape[0]
        time_steps = input_shape[1]

        if return_sequences:
            return (batch_size, time_steps, units)
        else:
            return (batch_size, units)

    def _handle_embedding(self, input_shape, params):
        logger.debug(f"_handle_embedding - input_shape: {input_shape}, params: {params}")
        output_dim = params['output_dim']
        # Handle dictionary values in output_dim
        if isinstance(output_dim, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in output_dim:
                output_dim = output_dim['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_embedding - output_dim is a dict without 'value' key: {output_dim}, using default")
                output_dim = 128  # Default value
        logger.debug(f"_handle_embedding - output_dim after processing: {output_dim}")
        # Embedding adds an extra dimension: (batch, sequence_length, output_dim)
        return (input_shape[0], input_shape[1], output_dim)

    def _handle_globalaveragepooling2d(self, input_shape, params):
        logger.debug(f"_handle_globalaveragepooling2d - input_shape: {input_shape}, params: {params}")
        data_format = params.get('data_format', 'channels_last')
        if data_format == 'channels_first':
            batch, channels, h, w = input_shape
            try:
                return (batch, channels)
            except (TypeError, ValueError):
                logger.debug(f"_handle_globalaveragepooling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], 1)  # Default value
        else:
            batch, h, w, channels = input_shape
            try:
                return (batch, channels)
            except (TypeError, ValueError):
                logger.debug(f"_handle_globalaveragepooling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], 1)  # Default value

    def _handle_upsampling2d(self, input_shape, params):
        logger.debug(f"_handle_upsampling2d - input_shape: {input_shape}, params: {params}")
        size = params.get('size', (2, 2))
        # Handle dictionary values in size
        if isinstance(size, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in size:
                size_value = size['value']
                if isinstance(size_value, int):
                    size = (size_value, size_value)
                else:
                    size = (2, 2)  # Default value
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_upsampling2d - size is a dict without 'value' key: {size}, using default")
                size = (2, 2)  # Default value

        print(f"DEBUG: _handle_upsampling2d - size after processing: {size}")

        # Calculate new spatial dimensions, handling None
        if data_format == 'channels_last':
            # TensorFlow: input_shape = (batch, height, width, channels)
            if len(input_shape) >= 4:
                h, w = input_shape[1], input_shape[2]
                new_height = h * size[0] if h is not None else None
                new_width = w * size[1] if w is not None else None
                return (input_shape[0], new_height, new_width, input_shape[3])
            else:
                logger.debug(f"_handle_upsampling2d - Invalid input shape: {input_shape}, using default")
                return input_shape
        else:
            # PyTorch: input_shape = (batch, channels, height, width)
            if len(input_shape) >= 4:
                h, w = input_shape[2], input_shape[3]
                new_height = h * size[0] if h is not None else None
                new_width = w * size[1] if w is not None else None
                return (input_shape[0], input_shape[1], new_height, new_width)
            else:
                print(f"DEBUG: _handle_upsampling2d - Invalid input shape: {input_shape}, using default")
                return input_shape

    def _handle_multiheadattention(self, input_shape, params):
        print(f"DEBUG: _handle_multiheadattention - input_shape: {input_shape}, params: {params}")
        # MultiHeadAttention preserves the input shape: (batch, seq_len, d_model)
        return input_shape
    
    # Handle default helper
    def _handle_default(self, input_shape, params):
        # Default handler for unsupported layers
        return input_shape

    ### Padding detection, extraction and calculation ###
    def _calculate_padding(self, params, spatial_dims, kernel_size=None, stride=None):
        """Calculates padding based on provided parameters and input dimensions.

        This method handles different padding types: integer, list, tuple, or string.
        It returns the appropriate padding value based on the input.

        Args:
            params (dict): Layer parameters containing padding information.
            spatial_dims (tuple): Spatial dimensions of input (height, width).
            kernel_size (tuple, optional): Kernel size for 'same' padding calculation.
            stride (tuple, optional): Stride for 'same' padding calculation.

        Returns:
            int or tuple: Calculated padding value.
        """
        print(f"DEBUG: _calculate_padding - params: {params}, spatial_dims: {spatial_dims}")
        padding = params.get('padding', 0)

        # Handle dictionary values in padding
        if isinstance(padding, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in padding_mode:
                padding_mode = padding_mode['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_calculate_padding - padding is a dict without 'value' key: {padding_mode}, using default")
                padding_mode = 'valid'  # Default value

        if isinstance(padding, int):
            return padding
        elif isinstance(padding, (list, tuple)):
            return tuple(padding)
        elif padding == 'same':
            # For 'same' padding with stride > 1, calculate padding to maintain ceil(input/stride)
            if kernel_size is None:
                kernel_size = params.get('kernel_size', 3)
            if stride is None:
                stride = params.get('stride', 1)
            
            # Normalize kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            elif isinstance(kernel_size, dict):
                if 'value' in kernel_size:
                    kernel_value = kernel_size['value']
                    kernel_size = (kernel_value, kernel_value) if isinstance(kernel_value, int) else (3, 3)
                else:
                    kernel_size = (3, 3)
            
            # Normalize stride
            if isinstance(stride, int):
                stride = (stride, stride)
            elif isinstance(stride, (list, tuple)):
                stride = tuple(stride)
            else:
                stride = (1, 1)
            
            # Calculate padding for each dimension
            padding_list = []
            for i, (dim, k, s) in enumerate(zip(spatial_dims, kernel_size, stride)):
                if dim is None:
                    # For None dimensions, use kernel-based padding
                    p = (k - 1) // 2
                else:
                    # Calculate padding to achieve output_size = ceil(input_size / stride)
                    # Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1
                    # For same padding: output_size = ceil(input_size / stride)
                    output_size = (dim + s - 1) // s  # This is ceil(dim / s)
                    total_padding = max(0, (output_size - 1) * s + k - dim)
                    p = total_padding // 2
                padding_list.append(p)
            
            return tuple(padding_list)
        elif padding == 'valid':
            return 0
        elif isinstance(padding_mode, int):
            return padding_mode
        elif isinstance(padding_mode, (tuple, list)):
            return padding_mode[0]
        else:
            # Unknown padding type, default to 0
            return 0

    ### Layers Shape Propagation Visualization ###
    # NOTE: Duplicate methods removed - using earlier definitions

        # Add shape dimensions as bar chart
        shapes = [str(s[1]) for s in self.shape_history]
        # Calculate parameter count, handling None values (batch dimension)
        param_counts = []
        for shape in self.shape_history:
            # Replace None with 1 for calculation, then exclude batch dimension
            calc_shape = tuple(1 if dim is None else dim for dim in shape[1])
            if len(calc_shape) > 1:
                # Exclude batch dimension (first element) for parameter count
                param_count = np.prod(calc_shape[1:])
            else:
                param_count = np.prod(calc_shape)
            param_counts.append(param_count)

        fig.add_trace(go.Bar(
            x=[s[0] for s in self.shape_history],
            y=param_counts,
            text=shapes,
            name='Parameter Count'
        ))

        fig.update_layout(
            title='Network Shape Propagation',
            xaxis_title='Layer',
            yaxis_title='Parameters',
            template='plotly_white'
        )

        # Detect shape issues and optimization opportunities
        self.issues = detect_shape_issues(self.shape_history)
        self.optimizations = suggest_optimizations(self.shape_history)

        return {
            'dot_graph': self.dot,
            'plotly_chart': fig,
            'shape_history': self.shape_history,
            'issues': self.issues,
            'optimizations': self.optimizations
        }

    def get_shape_data(self):
        """Returns shape history as JSON."""
        import json
        return json.dumps([
            {"layer": layer[0], "output_shape": layer[1]}
            for layer in self.shape_history
        ])

    def get_layer_documentation(self, layer_type):
        """Get documentation for a specific layer type.

        Args:
            layer_type: Type of layer to get documentation for

        Returns:
            Dictionary with layer documentation
        """
        return get_layer_documentation(layer_type)

    def format_layer_documentation(self, layer_type):
        """Format documentation for a specific layer type as a readable string.

        Args:
            layer_type: Type of layer to format documentation for

        Returns:
            Formatted documentation string
        """
        return format_layer_documentation(layer_type)

    def detect_issues(self):
        """Detect potential issues in the model architecture.

        Returns:
            List of detected issues
        """
        self.issues = detect_shape_issues(self.shape_history)
        return self.issues

    def suggest_optimizations(self):
        """Suggest optimizations for the model architecture.

        Returns:
            List of optimization suggestions
        """
        self.optimizations = suggest_optimizations(self.shape_history)
        return self.optimizations

    def generate_interactive_visualization(self):
        """Generate an interactive HTML visualization of the model architecture.

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise DependencyError(
                dependency="plotly",
                feature="interactive visualization",
                install_hint="pip install plotly"
            )
        
        from plotly.subplots import make_subplots

        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Tensor Dimensions", "Memory Usage"),
                            specs=[[{"type": "scatter"}], [{"type": "bar"}]])

        # Add tensor dimension trace
        layer_names = [layer[0] for layer in self.shape_history]
        tensor_sizes = [np.prod([dim for dim in shape if dim is not None])
                       for _, shape in self.shape_history]

        fig.add_trace(
            go.Scatter(x=layer_names, y=tensor_sizes, mode='lines+markers', name='Tensor Size'),
            row=1, col=1
        )

        # Add memory usage trace
        memory_usage = [calculate_memory_usage(shape) / (1024 * 1024)
                       for _, shape in self.shape_history]

        fig.add_trace(
            go.Bar(x=layer_names, y=memory_usage, name='Memory (MB)'),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(height=800, title_text="Model Shape Analysis")

        return fig

    def export_visualization(self, format='html'):
        """Export visualization to various formats.

        Args:
            format: Output format ('html', 'png', or 'mermaid')

        Returns:
            Visualization in the specified format
        """
        if format == 'html':
            fig = self.generate_interactive_visualization()
            return fig.to_html()
        elif format == 'png':
            fig = self.generate_interactive_visualization()
            return fig.to_image(format='png')
        elif format == 'mermaid':
            # Generate mermaid.js flowchart
            mermaid = "graph TD\n"
            for i, (layer_name, shape) in enumerate(self.shape_history):
                mermaid += f"  L{i}[{layer_name}<br>{shape}]\n"

            for from_id, to_id in self.layer_connections:
                mermaid += f"  L{from_id} --> L{to_id}\n"

            return mermaid
        else:
            raise ValueError(f"Unsupported format: {format}")

    def propagate_model(self, input_shapes, model_def):
        """Propagate shapes through a complete model with multiple inputs/outputs.

        Args:
            input_shapes: Dictionary mapping input names to shapes
            model_def: Model definition with layers, inputs, and outputs

        Returns:
            Dictionary mapping output names to shapes
        """
        # Track shapes by layer name for reference by other layers
        shape_map = {input_name: shape for input_name, shape in input_shapes.items()}

        # Process each layer in topological order
        for layer in model_def.get('layers', []):
            layer_name = layer.get('name')
            if not layer_name:
                continue

            # Get input shapes (could be multiple)
            layer_input = layer.get('input')
            if isinstance(layer_input, list):
                input_shapes_list = [shape_map[input_name] for input_name in layer_input
                              if input_name in shape_map]

                # Handle merging of inputs based on layer type
                if layer['type'] == 'Concatenate':
                    input_shape = handle_concatenate(input_shapes_list, layer.get('params', {}))
                elif layer['type'] == 'Add':
                    input_shape = handle_add(input_shapes_list, layer.get('params', {}))
                else:
                    # Default to first input shape if we don't know how to merge
                    input_shape = input_shapes_list[0] if input_shapes_list else None
            else:
                input_shape = shape_map.get(layer_input)

            if input_shape is None:
                continue

            # Propagate through this layer
            output_shape = self.propagate(input_shape, layer, model_def.get('framework', 'tensorflow'))

            # Store output shape
            shape_map[layer_name] = output_shape

            # Add to shape history
            self._visualize_layer(layer['type'], output_shape)

            # Add connection if we have previous layers
            if isinstance(layer_input, list):
                for input_name in layer_input:
                    if input_name in shape_map:
                        # Find the index of the input layer in shape_history
                        for i, (hist_name, _) in enumerate(self.shape_history):
                            if hist_name == input_name:
                                self._create_connection(i, len(self.shape_history) - 1)
                                break
            elif layer_input in shape_map:
                # Find the index of the input layer in shape_history
                for i, (hist_name, _) in enumerate(self.shape_history):
                    if hist_name == layer_input:
                        self._create_connection(i, len(self.shape_history) - 1)
                        break

        # Return shapes for all output layers
        return {output: shape_map[output] for output in model_def.get('outputs', [])
               if output in shape_map}

    def _log_shape(self, shape, stage):
        if self.debug:
            logging.info(f"{stage.upper()} SHAPE: {shape}")
            logging.debug(f"Shape details: {self._shape_analysis(shape)}")

    def _shape_analysis(self, shape):
        return {
            'total_parameters': np.prod([d for d in shape if d]),
            'spatial_dims': shape[2:-1] if len(shape) > 2 else None,
            'channel_dim': shape[1] if len(shape) > 1 else None
        }

    ### Loading Pretrained Models ####

    def load_pretrained(self, model_name, pretrained=True):
        if self.hub is None:
            raise DependencyError(
                dependency='triton, huggingface_hub',
                feature='PretrainedModelHub',
                install_hint='pip install neural-dsl[full]'
            )
        model = self.hub.load(model_name, pretrained)
        # Propagate shapes through pretrained model
        input_shape = (1, 3, 224, 224)  # Default for ResNet50
        for layer in model.layers:
            input_shape = self.propagate(input_shape, layer, "pytorch")

### Shape Validation for Error Handling ###

class ShapeValidator:
    @staticmethod
    def validate_layer(layer_type, input_shape, params):
        validators = {
            'Conv2D': lambda: ShapeValidator._validate_conv(input_shape, params),
            'Dense': lambda: ShapeValidator._validate_dense(input_shape, params)
        }

        if validator := validators.get(layer_type):
            validator()

    @staticmethod
    def _validate_conv(input_shape, params):
        if len(input_shape) != 4:
            raise ShapeMismatchError(
                f"Conv layers need 4D input. Got {len(input_shape)}D",
                input_shape=input_shape,
                layer_type='Conv2D'
            )
        if params['kernel_size'] > input_shape[2]:
            raise ShapeMismatchError(
                f"Kernel size {params['kernel_size']} exceeds input dimension {input_shape[2]}",
                input_shape=input_shape,
                layer_type='Conv2D'
            )

    @staticmethod
    def _validate_dense(input_shape, params):
        if len(input_shape) > 2:
            raise ShapeMismatchError(
                f"Dense layer expects 2D input (batch, features). Got {len(input_shape)}D: {input_shape}",
                input_shape=input_shape,
                layer_type='Dense'
            )
# Unified parameter handling for TF/PyTorch
FRAMEWORK_DEFAULTS = {
    'tensorflow': {
        'data_format': 'channels_last',
        'padding': 'same'
    },
    'pytorch': {
        'data_format': 'channels_first',
        'padding': 0
    }
}

def get_framework_params(framework):
    return FRAMEWORK_DEFAULTS.get(framework.lower(), FRAMEWORK_DEFAULTS['tensorflow'])

### Real-Time Shape Visualization ###

def _calculate_shape(self, input_shape, layer):
    if layer["type"] == "Dense":
        return (input_shape[0], layer["params"]["units"])
    elif layer["type"] == "Conv2D":
        return (input_shape[0], input_shape[1], input_shape[2], layer["params"]["filters"])
    elif layer["type"] == "Flatten":
        return (input_shape[0], np.prod(input_shape[1:]))
    return input_shape

### Compute FLOPs and memory usage for visualization ###
def compute_flops_params(layer, input_shape):
    """Estimate FLOPs and parameter counts for a given layer."""
    if layer["type"] == "Dense":
        units = layer["params"]["units"]
        params = input_shape[1] * units + units  # Weights + biases
        flops = 2 * params  # Two operations per weight (multiply + add)

    elif layer["type"] == "Conv2D":
        filters = layer["params"]["filters"]
        kernel_size = layer["params"]["kernel_size"]
        stride = layer["params"].get("stride", 1)
        params = (kernel_size[0] * kernel_size[1] * input_shape[-1] + 1) * filters
        output_height = (input_shape[1] - kernel_size[0]) // stride + 1
        output_width = (input_shape[2] - kernel_size[1]) // stride + 1
        flops = params * output_height * output_width

    return params, flops

#######################################
### Gradient Flow Visualization #######
#######################################
def register_gradient_hooks(model):
    """Attaches hooks to capture gradient magnitudes during backprop."""
    gradient_trace = []

    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_norm = grad_output[0].detach().abs().mean().item()
            gradient_trace.append({"layer": module.__class__.__name__, "grad_norm": grad_norm})

    for layer in model.children():
        layer.register_backward_hook(hook)

    return gradient_trace

#####################################
### Dead Neurons Detection ##########
#####################################
def detect_dead_neurons(layer, input, output):
    """Detects inactive neurons (dead neurons)."""
    # Return early if torch is not available
    if not TORCH_AVAILABLE or torch is None:
        return {"layer": layer.__class__.__name__, "dead_ratio": 0.0, "error": "torch not available"}
    dead_neurons = (output.detach() == 0).sum().item()
    total_neurons = output.numel()
    dead_ratio = dead_neurons / total_neurons

    return {"layer": layer.__class__.__name__, "dead_ratio": dead_ratio}

######################################
### Activation Anomalies Detection ###
######################################
def detect_activation_anomalies(layer, input, output):
    """Flags NaNs, extremely high activations, or overflows."""
    # Return early if torch is not available
    if not TORCH_AVAILABLE or torch is None:
        return {
            "layer": layer.__class__.__name__,
            "mean_activation": 0.0,
            "anomaly": False,
            "error": "torch not available"
        }
    mean_activation = output.detach().abs().mean().item()
    # torch is guaranteed to be available here due to the check above
    has_nan = torch.isnan(output).sum().item() > 0
    is_exploding = mean_activation > 100  # Threshold for detecting abnormally high activations

    return {
        "layer": layer.__class__.__name__,
        "mean_activation": mean_activation,
        "anomaly": has_nan or is_exploding
    }


######################
### Step Debugging ###
######################
def step_debug_hook(module, input, output):
    """Pauses execution at this layer for manual debugging."""
    print(f"Paused at layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}, Output shape: {output.shape}")

    # Wait for user input before continuing
    input("Press Enter to continue...")
