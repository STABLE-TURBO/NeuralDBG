"""
Example script demonstrating model export and deployment workflows.
"""

import os
import numpy as np
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.export import ModelExporter


def example_onnx_export():
    """Export a model to ONNX format with optimization."""
    print("=" * 60)
    print("Example 1: ONNX Export with Optimization")
    print("=" * 60)
    
    model_dsl = """
network MyClassifier {
    input: shape(1, 28, 28)
    layers: [
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=128, activation="relu"),
        Dropout(rate=0.5),
        Output(units=10, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    exporter = ModelExporter(model_data, backend='tensorflow')
    
    output_path = exporter.export_onnx(
        output_path='examples/my_classifier.onnx',
        opset_version=13,
        optimize=True
    )
    
    print(f"✓ Model exported to: {output_path}")
    print(f"✓ Optimization passes applied")
    print()


def example_tflite_quantized_export():
    """Export a model to TensorFlow Lite with quantization."""
    print("=" * 60)
    print("Example 2: TensorFlow Lite Export with Int8 Quantization")
    print("=" * 60)
    
    model_dsl = """
network MobileNet {
    input: shape(3, 224, 224)
    layers: [
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=1000, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    exporter = ModelExporter(model_data, backend='tensorflow')
    
    def representative_dataset():
        """Generate representative data for quantization calibration."""
        for _ in range(100):
            yield [np.random.randn(1, 3, 224, 224).astype(np.float32)]
    
    output_path = exporter.export_tflite(
        output_path='examples/mobilenet_int8.tflite',
        quantize=True,
        quantization_type='int8',
        representative_dataset=representative_dataset
    )
    
    print(f"✓ Model exported to: {output_path}")
    print(f"✓ Int8 quantization applied (4x size reduction)")
    print()


def example_torchscript_export():
    """Export a PyTorch model to TorchScript."""
    print("=" * 60)
    print("Example 3: TorchScript Export")
    print("=" * 60)
    
    model_dsl = """
network ResNetBlock {
    input: shape(3, 32, 32)
    layers: [
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=512, activation="relu"),
        Output(units=10, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    exporter = ModelExporter(model_data, backend='pytorch')
    
    output_path = exporter.export_torchscript(
        output_path='examples/resnet_block.pt',
        method='trace'
    )
    
    print(f"✓ Model exported to: {output_path}")
    print(f"✓ TorchScript trace method used")
    print()


def example_tensorflow_serving_deployment():
    """Export and prepare a model for TensorFlow Serving."""
    print("=" * 60)
    print("Example 4: TensorFlow Serving Deployment")
    print("=" * 60)
    
    model_dsl = """
network ProductionModel {
    input: shape(1, 28, 28)
    layers: [
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=128, activation="relu"),
        Output(units=10, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    exporter = ModelExporter(model_data, backend='tensorflow')
    
    model_path = exporter.export_savedmodel(
        output_path='examples/production_model_saved'
    )
    
    print(f"✓ SavedModel exported to: {model_path}")
    
    config_path = exporter.create_tfserving_config(
        model_path=model_path,
        model_name='production_model',
        output_dir='examples/tfserving_deployment',
        version=1
    )
    
    print(f"✓ TensorFlow Serving config created: {config_path}")
    
    scripts = exporter.generate_deployment_scripts(
        output_dir='examples/tfserving_deployment',
        deployment_type='tfserving'
    )
    
    print(f"✓ Deployment scripts generated:")
    for script in scripts:
        print(f"  - {script}")
    print()
    print("To deploy:")
    print("  cd examples/tfserving_deployment")
    print("  ./start_tfserving.sh")
    print()


def example_torchserve_deployment():
    """Export and prepare a model for TorchServe."""
    print("=" * 60)
    print("Example 5: TorchServe Deployment")
    print("=" * 60)
    
    model_dsl = """
network ImageClassifier {
    input: shape(3, 224, 224)
    layers: [
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=512, activation="relu"),
        Dropout(rate=0.5),
        Output(units=1000, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    exporter = ModelExporter(model_data, backend='pytorch')
    
    model_path = exporter.export_torchscript(
        output_path='examples/image_classifier.pt',
        method='trace'
    )
    
    print(f"✓ TorchScript model exported to: {model_path}")
    
    config_path, model_store = exporter.create_torchserve_config(
        model_path=model_path,
        model_name='image_classifier',
        output_dir='examples/torchserve_deployment',
        handler='image_classifier',
        batch_size=8,
        max_batch_delay=100
    )
    
    print(f"✓ TorchServe config created: {config_path}")
    print(f"✓ Model store: {model_store}")
    
    scripts = exporter.generate_deployment_scripts(
        output_dir='examples/torchserve_deployment',
        deployment_type='torchserve'
    )
    
    print(f"✓ Deployment scripts generated:")
    for script in scripts:
        print(f"  - {script}")
    print()
    print("To deploy:")
    print("  1. Create MAR file:")
    print("     torch-model-archiver --model-name image_classifier \\")
    print("       --version 1.0 --serialized-file examples/image_classifier.pt \\")
    print("       --handler image_classifier \\")
    print("       --export-path examples/torchserve_deployment/model-store")
    print("  2. Start TorchServe:")
    print("     cd examples/torchserve_deployment")
    print("     ./start_torchserve.sh")
    print()


def example_multi_format_export():
    """Export a model to multiple formats simultaneously."""
    print("=" * 60)
    print("Example 6: Multi-Format Export")
    print("=" * 60)
    
    model_dsl = """
network UniversalModel {
    input: shape(1, 28, 28)
    layers: [
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=64, activation="relu"),
        Output(units=10, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    print("Exporting to multiple formats:")
    
    exporter_tf = ModelExporter(model_data, backend='tensorflow')
    onnx_path = exporter_tf.export_onnx('examples/universal_model.onnx', optimize=True)
    print(f"  ✓ ONNX: {onnx_path}")
    
    tflite_path = exporter_tf.export_tflite('examples/universal_model.tflite', quantize=True, quantization_type='dynamic')
    print(f"  ✓ TFLite: {tflite_path}")
    
    savedmodel_path = exporter_tf.export_savedmodel('examples/universal_model_saved')
    print(f"  ✓ SavedModel: {savedmodel_path}")
    
    exporter_pt = ModelExporter(model_data, backend='pytorch')
    torchscript_path = exporter_pt.export_torchscript('examples/universal_model.pt')
    print(f"  ✓ TorchScript: {torchscript_path}")
    
    print()
    print("✓ Model exported to 4 different formats for maximum compatibility")
    print()


def main():
    """Run all examples."""
    os.makedirs('examples', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Neural DSL Model Export and Deployment Examples")
    print("=" * 60 + "\n")
    
    try:
        example_onnx_export()
    except Exception as e:
        print(f"✗ Example 1 failed: {e}\n")
    
    try:
        example_tflite_quantized_export()
    except Exception as e:
        print(f"✗ Example 2 failed: {e}\n")
    
    try:
        example_torchscript_export()
    except Exception as e:
        print(f"✗ Example 3 failed: {e}\n")
    
    try:
        example_tensorflow_serving_deployment()
    except Exception as e:
        print(f"✗ Example 4 failed: {e}\n")
    
    try:
        example_torchserve_deployment()
    except Exception as e:
        print(f"✗ Example 5 failed: {e}\n")
    
    try:
        example_multi_format_export()
    except Exception as e:
        print(f"✗ Example 6 failed: {e}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor detailed deployment guides, see docs/deployment.md")


if __name__ == '__main__':
    main()
