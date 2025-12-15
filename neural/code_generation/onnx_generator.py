import logging
from typing import Any, Dict

from neural.code_generation.base_generator import BaseCodeGenerator


logger = logging.getLogger(__name__)


class ONNXGenerator(BaseCodeGenerator):
    def generate(self) -> str:
        return export_onnx(self.model_data, "model.onnx")

    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        pass


def generate_onnx(model_data: Dict[str, Any]):
    """Generate ONNX model"""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    initializers = []
    value_infos = []
    current_input = "input"
    current_shape = list(model_data["input"]["shape"])

    for i, layer in enumerate(model_data['layers']):
        layer_type = layer['type']
        params = layer.get('params', {})
        output_name = f"layer_{i}_output"

        if layer_type == "Conv2D":
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', [3, 3])
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            strides = params.get('strides', [1, 1])
            if isinstance(strides, int):
                strides = [strides, strides]

            # Determine input channels from current_shape
            # For channels_last (H, W, C), channels are at index -1
            # For channels_first (C, H, W), channels are at index 0
            if len(current_shape) == 3:
                # Assume channels_last: (H, W, C)
                in_channels = current_shape[-1]
            elif len(current_shape) == 4:
                # With batch: (N, H, W, C) channels_last
                in_channels = current_shape[-1]
            else:
                in_channels = 3  # Default fallback

            # Create weight tensor (filters, in_channels, kernel_h, kernel_w)
            weight_name = f"conv_{i}_weight"
            weight_shape = [filters, in_channels, kernel_size[0], kernel_size[1]]
            weight_data = np.random.randn(*weight_shape).astype(np.float32) * 0.01
            weight_tensor = numpy_helper.from_array(weight_data, name=weight_name)
            initializers.append(weight_tensor)

            # Create bias tensor
            bias_name = f"conv_{i}_bias"
            bias_data = np.zeros([filters], dtype=np.float32)
            bias_tensor = numpy_helper.from_array(bias_data, name=bias_name)
            initializers.append(bias_tensor)

            nodes.append(helper.make_node(
                'Conv',
                inputs=[current_input, weight_name, bias_name],
                outputs=[output_name],
                kernel_shape=kernel_size,
                strides=strides
            ))

            # Update shape: Conv2D changes channels dimension
            if len(current_shape) == 3:
                # (H, W, C) -> compute new H, W
                new_h = current_shape[0] - kernel_size[0] + 1
                new_w = current_shape[1] - kernel_size[1] + 1
                current_shape = [new_h, new_w, filters]
            elif len(current_shape) == 4:
                # (N, H, W, C)
                new_h = current_shape[1] - kernel_size[0] + 1
                new_w = current_shape[2] - kernel_size[1] + 1
                current_shape = [current_shape[0], new_h, new_w, filters]

            # Add value_info for intermediate tensor
            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "MaxPooling2D":
            pool_size = params.get('pool_size', [2, 2])
            if isinstance(pool_size, int):
                pool_size = [pool_size, pool_size]

            nodes.append(helper.make_node(
                'MaxPool',
                inputs=[current_input],
                outputs=[output_name],
                kernel_shape=pool_size,
                strides=pool_size
            ))

            # Update shape for pooling
            if len(current_shape) == 3:
                current_shape = [current_shape[0] // pool_size[0],
                               current_shape[1] // pool_size[1],
                               current_shape[2]]
            elif len(current_shape) == 4:
                current_shape = [current_shape[0],
                               current_shape[1] // pool_size[0],
                               current_shape[2] // pool_size[1],
                               current_shape[3]]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "Flatten":
            nodes.append(helper.make_node(
                'Flatten',
                inputs=[current_input],
                outputs=[output_name],
                axis=1
            ))

            # Update shape: flatten to (batch, features)
            if len(current_shape) > 1:
                if current_shape[0] is None:
                    features = 1
                    for dim in current_shape[1:]:
                        if dim is not None:
                            features *= dim
                    current_shape = [None, features]
                else:
                    features = 1
                    for dim in current_shape:
                        if dim is not None:
                            features *= dim
                    current_shape = [1, features]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "Dense":
            units = params.get('units', 64)

            # Determine input features
            if isinstance(current_shape[-1], int):
                in_features = current_shape[-1]
            else:
                in_features = 128  # Default fallback

            # Create weight tensor
            weight_name = f"dense_{i}_weight"
            weight_shape = [units, in_features]
            weight_data = np.random.randn(*weight_shape).astype(np.float32) * 0.01
            weight_tensor = numpy_helper.from_array(weight_data, name=weight_name)
            initializers.append(weight_tensor)

            # Create bias tensor
            bias_name = f"dense_{i}_bias"
            bias_data = np.zeros([units], dtype=np.float32)
            bias_tensor = numpy_helper.from_array(bias_data, name=bias_name)
            initializers.append(bias_tensor)

            nodes.append(helper.make_node(
                'Gemm',
                inputs=[current_input, weight_name, bias_name],
                outputs=[output_name],
                alpha=1.0,
                beta=1.0,
                transB=1
            ))

            # Update shape
            current_shape = [current_shape[0], units]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "MultiHeadAttention":
            num_heads = params.get('num_heads', 8)
            nodes.append(helper.make_node(
                'Attention',
                inputs=[current_input, current_input, current_input],
                outputs=[output_name],
                num_heads=num_heads
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "Dropout":
            nodes.append(helper.make_node(
                'Dropout',
                inputs=[current_input],
                outputs=[output_name],
                ratio=params.get('rate', 0.5)
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "Embedding":
            nodes.append(helper.make_node(
                'Gather',
                inputs=[current_input],
                outputs=[output_name]
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "GlobalAveragePooling1D":
            nodes.append(helper.make_node(
                'GlobalAveragePool',
                inputs=[current_input],
                outputs=[output_name]
            ))

            # Update shape: reduces spatial dimensions
            if len(current_shape) > 2:
                current_shape = [current_shape[0], current_shape[-1]]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "GlobalAveragePooling2D":
            nodes.append(helper.make_node(
                'GlobalAveragePool',
                inputs=[current_input],
                outputs=[output_name]
            ))

            # Update shape: reduces spatial dimensions
            if len(current_shape) > 2:
                current_shape = [current_shape[0], current_shape[-1]]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type in ["GlobalMaxPooling1D", "GlobalMaxPooling2D"]:
            nodes.append(helper.make_node(
                'GlobalMaxPool',
                inputs=[current_input],
                outputs=[output_name]
            ))

            # Update shape: reduces spatial dimensions
            if len(current_shape) > 2:
                current_shape = [current_shape[0], current_shape[-1]]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "TransformerEncoder":
            num_heads = params.get('num_heads', 8)
            nodes.append(helper.make_node(
                'MultiHeadAttention',
                inputs=[current_input, current_input, current_input],
                outputs=[output_name],
                num_heads=num_heads
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "TransformerDecoder":
            num_heads = params.get('num_heads', 8)
            self_attn_output = f"layer_{i}_self_attn"
            cross_attn_output = f"layer_{i}_cross_attn"
            nodes.append(helper.make_node(
                'MultiHeadAttention',
                inputs=[current_input, current_input, current_input],
                outputs=[self_attn_output],
                num_heads=num_heads
            ))

            value_infos.append(helper.make_tensor_value_info(
                self_attn_output, TensorProto.FLOAT, current_shape
            ))

            nodes.append(helper.make_node(
                'MultiHeadAttention',
                inputs=[self_attn_output, 'encoder_output', 'encoder_output'],
                outputs=[cross_attn_output],
                num_heads=num_heads
            ))

            value_infos.append(helper.make_tensor_value_info(
                cross_attn_output, TensorProto.FLOAT, current_shape
            ))

            nodes.append(helper.make_node(
                'Identity',
                inputs=[cross_attn_output],
                outputs=[output_name]
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "BatchNormalization":
            # Get number of channels
            num_channels = current_shape[-1] if len(current_shape) > 0 else 64

            # Create scale (gamma), bias (beta), mean, and var tensors
            scale_name = f"bn_{i}_scale"
            bias_name = f"bn_{i}_bias"
            mean_name = f"bn_{i}_mean"
            var_name = f"bn_{i}_var"

            scale_data = np.ones([num_channels], dtype=np.float32)
            bias_data = np.zeros([num_channels], dtype=np.float32)
            mean_data = np.zeros([num_channels], dtype=np.float32)
            var_data = np.ones([num_channels], dtype=np.float32)

            initializers.append(numpy_helper.from_array(scale_data, name=scale_name))
            initializers.append(numpy_helper.from_array(bias_data, name=bias_name))
            initializers.append(numpy_helper.from_array(mean_data, name=mean_name))
            initializers.append(numpy_helper.from_array(var_data, name=var_name))

            nodes.append(helper.make_node(
                'BatchNormalization',
                inputs=[current_input, scale_name, bias_name, mean_name, var_name],
                outputs=[output_name]
            ))

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        elif layer_type == "Output":
            units = params.get('units', 10)

            # Determine input features
            if isinstance(current_shape[-1], int):
                in_features = current_shape[-1]
            else:
                in_features = 128  # Default fallback

            # Create weight tensor
            weight_name = f"output_{i}_weight"
            weight_shape = [units, in_features]
            weight_data = np.random.randn(*weight_shape).astype(np.float32) * 0.01
            weight_tensor = numpy_helper.from_array(weight_data, name=weight_name)
            initializers.append(weight_tensor)

            # Create bias tensor
            bias_name = f"output_{i}_bias"
            bias_data = np.zeros([units], dtype=np.float32)
            bias_tensor = numpy_helper.from_array(bias_data, name=bias_name)
            initializers.append(bias_tensor)

            nodes.append(helper.make_node(
                'Gemm',
                inputs=[current_input, weight_name, bias_name],
                outputs=[output_name],
                alpha=1.0,
                beta=1.0,
                transB=1
            ))

            # Update shape
            current_shape = [current_shape[0], units]

            value_infos.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, current_shape
            ))

        current_input = output_name

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, model_data["input"]["shape"]
    )
    output_info = helper.make_tensor_value_info(
        current_input, TensorProto.FLOAT, current_shape
    )
    graph = helper.make_graph(
        nodes=nodes,
        name="NeuralModel",
        inputs=[input_info],
        outputs=[output_info],
        initializer=initializers,
        value_info=value_infos
    )

    model = helper.make_model(graph, producer_name="Neural")
    model.opset_import[0].version = 13

    return model


def export_onnx(
    model_data: Dict[str, Any],
    filename: str = "model.onnx",
    optimize: bool = True
) -> str:
    """Export model to ONNX format with optional optimization.
    
    Args:
        model_data: Dictionary containing model configuration
        filename: Output filename for the ONNX model
        optimize: Whether to apply ONNX optimizer passes
        
    Returns:
        Status message indicating the file was saved
    """
    import onnx
    model = generate_onnx(model_data)

    if optimize:
        try:
            from onnx import optimizer
            passes = [
                'eliminate_identity',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_bn_into_conv',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]
            model = optimizer.optimize(model, passes)
        except ImportError:
            logger.warning("ONNX optimizer not available, saving unoptimized model")

    onnx.save(model, filename)
    return f"ONNX model saved to {filename}"
