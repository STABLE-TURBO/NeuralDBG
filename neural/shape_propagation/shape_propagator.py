import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import psutil
from graphviz import Digraph

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
            'BatchNormalization': {'momentum': 'decay'}
        }

        # Initialize visualization
        self.dot = Digraph(comment='Neural Network Architecture')
        self.dot.attr('node', shape='record', style='filled', fillcolor='lightgrey')

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
        """Compute performance metrics (FLOPs, memory, execution time) for a layer."""
        layer_type = layer["type"]
        params = layer.get("params", {})

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

        # Compute memory usage (bytes)
        memory = calculate_memory_usage(input_shape, output_shape)

        # Simulate execution time (microseconds) - in real scenarios, this would be measured
        exec_time = flops / 1e6  # Simplified: 1 GFLOPS assumed
        comp_time = exec_time * 0.8  # 80% of exec_time is compute
        trans_time = exec_time * 0.2  # 20% is data transfer

        return flops, memory, exec_time, comp_time, trans_time

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
        label = f"{layer_type}\\nOutput: {output_shape}"
        self.dot.node(str(self.current_layer), label=label)
        self.current_layer += 1

    def _create_connection(self, from_layer: int, to_layer: int):
        """Create an edge between two layers."""
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

        return {
            'dot_graph': self.dot,
            'plotly_chart': fig,
            'shape_history': self.shape_history,
            'issues': self.issues,
            'optimizations': self.optimizations
        }

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

        padding = self._calculate_padding(params, input_shape[2] if data_format == 'channels_first' else input_shape[1])

        if isinstance(padding, int):
            padding = (padding,) * len(spatial_dims)
        elif isinstance(padding, (list, tuple)):
            padding = tuple(padding)
        elif isinstance(padding, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in padding:
                padding_value = padding['value']
                if isinstance(padding_value, int):
                    padding = (padding_value,) * len(spatial_dims)
                else:
                    padding = (0,) * len(spatial_dims)  # Default value
            # Otherwise, use a default value
            else:
                logger.debug(f"_handle_conv2d - padding is a dict without 'value' key: {padding}, using default")
                padding = (0,) * len(spatial_dims)  # Default value

        logger.debug(f"_handle_conv2d - kernel: {kernel}, stride: {stride}, padding: {padding}")

        output_spatial = [
            (dim + 2*pad - k) // stride + 1
            for dim, k, pad in zip(spatial_dims, kernel, padding)
        ]
        if any(dim <= 0 for dim in output_spatial):
            logger.debug(f"_handle_conv2d - Invalid Conv2D output dimensions: {output_spatial}, using default")
            output_spatial = [1, 1]  # Default value to avoid errors

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
        else:
            stride_h, stride_w = stride if isinstance(stride, (tuple, list)) else (pool_size[0], pool_size[0])

        logger.debug(f"_handle_maxpooling2d - pool_size: {pool_size}, stride_h: {stride_h}, stride_w: {stride_w}")

        if data_format == 'channels_first':
            batch, channels, h, w = input_shape
            try:
                h_out = (h - pool_size[0]) // stride_h + 1
                w_out = (w - pool_size[1]) // stride_w + 1
            except (TypeError, ValueError):
                logger.debug(f"_handle_maxpooling2d - Invalid input shape: {input_shape}, using default")
                h_out = w_out = 1  # Default value
            return (batch, channels, h_out, w_out)
        else:
            batch, h, w, channels = input_shape
            try:
                h_out = (h - pool_size[0]) // stride_h + 1
                w_out = (w - pool_size[1]) // stride_w + 1
            except (TypeError, ValueError):
                logger.debug(f"_handle_maxpooling2d - Invalid input shape: {input_shape}, using default")
                h_out = w_out = 1  # Default value
            return (batch, h_out, w_out, channels)

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
        logger.debug(f"_handle_upsampling2d - size after processing: {size}")
        data_format = params.get('data_format', 'channels_last')
        if data_format == 'channels_first':
            batch, channels, h, w = input_shape
            try:
                return (batch, channels, h * size[0], w * size[1])
            except (TypeError, ValueError):
                logger.debug(f"_handle_upsampling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], channels, 1, 1)  # Default value
        else:
            batch, h, w, channels = input_shape
            try:
                return (batch, h * size[0], w * size[1], channels)
            except (TypeError, ValueError):
                logger.debug(f"_handle_upsampling2d - Invalid input shape: {input_shape}, using default")
                return (input_shape[0], 1, 1, channels)  # Default value

    def _handle_multiheadattention(self, input_shape, params):
        logger.debug(f"_handle_multiheadattention - input_shape: {input_shape}, params: {params}")
        # MultiHeadAttention typically doesn't change the shape
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, d_model)
        return input_shape

############################
### Helper functions ###
############################

    def _calculate_padding(self, params, input_dim):
        """Calculate padding based on params."""
        logger.debug(f"_calculate_padding - params: {params}, input_dim: {input_dim}")
        padding_mode = params.get('padding', 'valid')

        if isinstance(padding_mode, dict):
            # If it's a dictionary with a 'value' key, use that value
            if 'value' in padding_mode:
                padding_mode = padding_mode['value']
            # Otherwise, use a default value
            else:
                logger.debug(f"_calculate_padding - padding is a dict without 'value' key: {padding_mode}, using default")
                padding_mode = 'valid'  # Default value

        if padding_mode == 'same':
            kernel = params.get('kernel_size', 3)
            if isinstance(kernel, dict):
                # If it's a dictionary with a 'value' key, use that value
                if 'value' in kernel:
                    kernel = kernel['value']
                # Otherwise, use a default value
                else:
                    logger.debug(f"_calculate_padding - kernel is a dict without 'value' key: {kernel}, using default")
                    kernel = 3  # Default value
            elif isinstance(kernel, (tuple, list)):
                kernel = kernel[0]
            elif not isinstance(kernel, int):
                logger.debug(f"_calculate_padding - Invalid kernel type: {type(kernel)}, value: {kernel}, using default")
                kernel = 3  # Default value
            return (kernel - 1) // 2
        elif padding_mode == 'valid':
            return 0
        elif isinstance(padding_mode, int):
            return padding_mode
        elif isinstance(padding_mode, (tuple, list)):
            return padding_mode[0]
        else:
            return 0

##############################################
### Hook Registration for Real Metrics ###
##############################################

def register_hooks(model, track_gradients=True, track_activations=True, track_dead_neurons=True):
    """
    Register forward and backward hooks on model layers.
    
    Args:
        model: PyTorch model
        track_gradients: Whether to track gradient flow
        track_activations: Whether to track activation statistics
        track_dead_neurons: Whether to detect dead neurons
    
    Returns:
        Dictionary mapping layer names to their metric data
    """
    metrics = {}
    
    def forward_hook(module, input, output):
        """Forward hook to capture activations."""
        layer_name = module.__class__.__name__
        if layer_name not in metrics:
            metrics[layer_name] = {}
        
        if track_activations:
            # Store activation statistics
            if isinstance(output, torch.Tensor):
                metrics[layer_name]['mean_activation'] = output.detach().cpu().mean().item()
                metrics[layer_name]['std_activation'] = output.detach().cpu().std().item()
                
                # Detect dead neurons (for ReLU-like activations)
                if track_dead_neurons and 'relu' in layer_name.lower():
                    total_neurons = output.numel()
                    dead_neurons = (output == 0).sum().item()
                    metrics[layer_name]['dead_ratio'] = dead_neurons / total_neurons if total_neurons > 0 else 0.0
    
    def backward_hook(module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        layer_name = module.__class__.__name__
        if layer_name not in metrics:
            metrics[layer_name] = {}
        
        if track_gradients:
            # Store gradient statistics
            if grad_output[0] is not None:
                metrics[layer_name]['grad_norm'] = grad_output[0].detach().cpu().norm().item()
    
    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module.register_forward_hook(forward_hook)
            if track_gradients:
                module.register_backward_hook(backward_hook)
    
    return metrics


def print_at_layer(layer_name):
    """
    Decorator to pause execution at a specific layer for debugging.
    
    Args:
        layer_name: Name of the layer to pause at
    
    Returns:
        Decorated forward hook function
    """
    def hook(module, input, output):
        if module.__class__.__name__ == layer_name:
            logger.info(f"Paused at layer: {module.__class__.__name__}")
            logger.info(f"Input shape: {input[0].shape}, Output shape: {output.shape}")
    return hook
