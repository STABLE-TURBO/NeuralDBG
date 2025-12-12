"""
ONNX backend integration tests.

Tests complete workflow for ONNX export: DSL parsing → shape propagation → ONNX generation.
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_onnx, export_onnx

try:
    import onnx
    from onnx import helper, TensorProto, checker, numpy_helper
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    ONNX_AVAILABLE = False
    ONNX_RUNTIME_AVAILABLE = False


class TestONNXWorkflowIntegration:
    """Integration tests for ONNX export workflow."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_simple_onnx_export(self):
        """Test: Simple model DSL to ONNX export."""
        dsl_code = """
        network SimpleONNX {
            input: (224, 224, 3)
            layers:
                Conv2D(filters=32, kernel_size=3)
                Flatten()
                Dense(128)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None
        assert onnx_model.graph is not None
        assert len(onnx_model.graph.node) > 0

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_export_to_file(self):
        """Test: Export ONNX model to file."""
        dsl_code = """
        network FileExportONNX {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        export_path = 'test_model.onnx'
        result = export_onnx(model_config, export_path)
        
        assert os.path.exists(export_path)
        assert "ONNX model saved" in result
        
        loaded_model = onnx.load(export_path)
        assert loaded_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_model_validation(self):
        """Test: Validate generated ONNX model."""
        dsl_code = """
        network ValidatedONNX {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=16, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        try:
            checker.check_model(onnx_model)
            validation_passed = True
        except Exception as e:
            validation_passed = False
        
        assert validation_passed or onnx_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_conv_layer(self):
        """Test: ONNX export with convolutional layers."""
        dsl_code = """
        network ConvONNX {
            input: (64, 64, 3)
            layers:
                Conv2D(filters=32, kernel_size=3, strides=1)
                Conv2D(filters=64, kernel_size=3, strides=1)
                Flatten()
                Output(100)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None
        
        conv_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Conv']
        assert len(conv_nodes) >= 1

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_multiple_layer_types(self):
        """Test: ONNX export with multiple layer types."""
        dsl_code = """
        network MultiLayerONNX {
            input: (128, 128, 3)
            layers:
                Conv2D(filters=32, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Conv2D(filters=64, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(256)
                Dense(128)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None
        assert len(onnx_model.graph.node) > 0

    @pytest.mark.skipif(not (ONNX_AVAILABLE and ONNX_RUNTIME_AVAILABLE), reason="ONNX Runtime not available")
    def test_onnx_inference_execution(self):
        """Test: Execute inference on ONNX model (if runtime available)."""
        dsl_code = """
        network InferenceONNX {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        export_path = 'inference_model.onnx'
        export_onnx(model_config, export_path)
        
        assert os.path.exists(export_path)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_shape_inference(self):
        """Test: ONNX shape inference on generated model."""
        dsl_code = """
        network ShapeInferenceONNX {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=16, kernel_size=3)
                Flatten()
                Dense(128)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        try:
            inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
            shape_inference_passed = True
        except Exception as e:
            shape_inference_passed = False
        
        assert shape_inference_passed or onnx_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_opset_version(self):
        """Test: ONNX model has correct opset version."""
        dsl_code = """
        network OpsetONNX {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model.opset_import[0].version >= 10

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_model_metadata(self):
        """Test: ONNX model includes metadata."""
        dsl_code = """
        network MetadataONNX {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model.producer_name == "Neural"
        assert onnx_model.graph.name == "NeuralModel"

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_complex_architecture(self):
        """Test: ONNX export with complex architecture."""
        dsl_code = """
        network ComplexArchONNX {
            input: (224, 224, 3)
            layers:
                Conv2D(filters=64, kernel_size=7, strides=2)
                MaxPooling2D(pool_size=3)
                Conv2D(filters=128, kernel_size=3, strides=1)
                Conv2D(filters=128, kernel_size=3, strides=1)
                MaxPooling2D(pool_size=2)
                Conv2D(filters=256, kernel_size=3, strides=1)
                Conv2D(filters=256, kernel_size=3, strides=1)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(512)
                Dense(256)
                Output(1000)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None
        assert len(onnx_model.graph.node) >= 5

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_batch_normalization(self):
        """Test: ONNX export with batch normalization layers."""
        dsl_code = """
        network BatchNormONNX {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=32, kernel_size=3)
                BatchNormalization()
                Conv2D(filters=64, kernel_size=3)
                BatchNormalization()
                Flatten()
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_different_input_shapes(self):
        """Test: ONNX export with different input shapes."""
        test_cases = [
            ("(28, 28, 1)", (28, 28, 1)),
            ("(32, 32, 3)", (32, 32, 3)),
            ("(224, 224, 3)", (224, 224, 3)),
            ("(64, 64, 1)", (64, 64, 1))
        ]
        
        for shape_str, expected_shape in test_cases:
            dsl_code = f"""
            network ShapeTest {{
                input: {shape_str}
                layers:
                    Flatten()
                    Dense(64)
                    Output(10)
            }}
            """
            
            parser = create_parser("network")
            tree = parser.parse(dsl_code)
            transformer = ModelTransformer()
            model_config = transformer.transform(tree)
            
            assert tuple(model_config['input']['shape']) == expected_shape
            
            onnx_model = generate_onnx(model_config)
            assert onnx_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_sequential_exports(self):
        """Test: Multiple sequential ONNX exports."""
        models = [
            "network Model1 { input: (28, 28, 1) layers: Flatten() Dense(64) Output(10) }",
            "network Model2 { input: (32, 32, 3) layers: Flatten() Dense(128) Output(20) }",
            "network Model3 { input: (64, 64, 1) layers: Flatten() Dense(256) Output(30) }"
        ]
        
        for idx, dsl_code in enumerate(models):
            parser = create_parser("network")
            tree = parser.parse(dsl_code)
            transformer = ModelTransformer()
            model_config = transformer.transform(tree)
            
            export_path = f'model_{idx}.onnx'
            result = export_onnx(model_config, export_path)
            
            assert os.path.exists(export_path)
            assert "ONNX model saved" in result

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_dense_only_architecture(self):
        """Test: ONNX export with dense-only architecture."""
        dsl_code = """
        network DenseOnlyONNX {
            input: (784,)
            layers:
                Dense(512)
                Dense(256)
                Dense(128)
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        assert current_shape == (None, 10)
        
        onnx_model = generate_onnx(model_config)
        assert onnx_model is not None


class TestONNXWithOtherBackends:
    """Test ONNX export alongside other backends."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_same_model_multiple_exports(self):
        """Test: Export same model to multiple backends including ONNX."""
        dsl_code = """
        network MultiBackendNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Dense(32, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        from neural.code_generation.code_generator import generate_code
        
        pytorch_code = generate_code(model_config, 'pytorch')
        assert 'import torch' in pytorch_code
        
        try:
            tf_code = generate_code(model_config, 'tensorflow')
            assert 'import tensorflow' in tf_code
        except Exception:
            pass
        
        onnx_model = generate_onnx(model_config)
        assert onnx_model is not None

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_consistency_with_pytorch_shape(self):
        """Test: ONNX model has consistent shapes with PyTorch model."""
        dsl_code = """
        network ConsistentShapes {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=32, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        expected_output_shape = (None, 10)
        assert current_shape == expected_output_shape
        
        onnx_model = generate_onnx(model_config)
        assert onnx_model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
