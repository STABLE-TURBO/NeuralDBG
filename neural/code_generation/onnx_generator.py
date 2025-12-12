from typing import Any, Dict
from neural.code_generation.base_generator import BaseCodeGenerator
import logging

logger = logging.getLogger(__name__)


class ONNXGenerator(BaseCodeGenerator):
    def generate(self) -> str:
        return export_onnx(self.model_data, "model.onnx")

    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        pass


def generate_onnx(model_data: Dict[str, Any]):
    """Generate ONNX model"""
    import onnx
    from onnx import helper, TensorProto
    
    nodes = []
    current_input = "input"

    for i, layer in enumerate(model_data['layers']):
        layer_type = layer['type']
        params = layer.get('params', {})
        output_name = f"layer_{i}_output"

        if layer_type == "Conv2D":
            nodes.append(helper.make_node(
                'Conv',
                inputs=[current_input],
                outputs=[output_name],
                kernel_shape=params.get('kernel_size', [3, 3]),
                strides=params.get('strides', [1, 1])
            ))

        current_input = output_name

    graph = helper.make_graph(
        nodes=nodes,
        name="NeuralModel",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, model_data["input"]["shape"])],
        outputs=[helper.make_tensor_value_info(current_input, TensorProto.FLOAT, None)],
        initializer=[]
    )

    model = helper.make_model(graph, producer_name="Neural")
    model.opset_import[0].version = 13

    return model


def export_onnx(model_data: Dict[str, Any], filename: str = "model.onnx", optimize: bool = True) -> str:
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
