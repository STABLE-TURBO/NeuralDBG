from typing import Any, Dict
from neural.code_generation.base_generator import BaseCodeGenerator


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


def export_onnx(model_data: Dict[str, Any], filename: str = "model.onnx") -> str:
    """Export model to ONNX format."""
    import onnx
    model = generate_onnx(model_data)
    onnx.save(model, filename)
    return f"ONNX model saved to {filename}"
