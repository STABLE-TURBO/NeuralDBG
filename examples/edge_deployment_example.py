"""
Example: Deploy a model for edge devices (mobile/IoT) with TensorFlow Lite.
This example demonstrates the full workflow from DSL to optimized edge deployment.
"""

import os
import numpy as np
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.export import ModelExporter


def create_mobile_optimized_model():
    """Create a lightweight model suitable for mobile deployment."""
    model_dsl = """
network MobileNetLite {
    input: shape(3, 224, 224)
    layers: [
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=512, activation="relu"),
        Dropout(rate=0.3),
        Output(units=1000, activation="softmax")
    ]
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""
    
    parser = create_parser('network')
    tree = parser.parse(model_dsl)
    model_data = ModelTransformer().transform(tree)
    
    return model_data


def export_for_mobile():
    """Export model for mobile deployment with various quantization options."""
    print("=" * 70)
    print("Mobile/Edge Deployment Example")
    print("=" * 70)
    print()
    
    model_data = create_mobile_optimized_model()
    exporter = ModelExporter(model_data, backend='tensorflow')
    
    os.makedirs('examples/mobile_models', exist_ok=True)
    
    # 1. Export baseline model (no quantization)
    print("1. Exporting baseline TFLite model (no quantization)...")
    baseline_path = exporter.export_tflite(
        output_path='examples/mobile_models/model_baseline.tflite',
        quantize=False
    )
    baseline_size = os.path.getsize(baseline_path)
    print(f"   ✓ Baseline model: {baseline_size / 1024:.2f} KB")
    print()
    
    # 2. Export with dynamic range quantization
    print("2. Exporting with dynamic range quantization...")
    dynamic_path = exporter.export_tflite(
        output_path='examples/mobile_models/model_dynamic.tflite',
        quantize=True,
        quantization_type='dynamic'
    )
    dynamic_size = os.path.getsize(dynamic_path)
    print(f"   ✓ Dynamic quantized: {dynamic_size / 1024:.2f} KB")
    print(f"   ✓ Size reduction: {(1 - dynamic_size/baseline_size) * 100:.1f}%")
    print()
    
    # 3. Export with float16 quantization
    print("3. Exporting with float16 quantization...")
    fp16_path = exporter.export_tflite(
        output_path='examples/mobile_models/model_fp16.tflite',
        quantize=True,
        quantization_type='float16'
    )
    fp16_size = os.path.getsize(fp16_path)
    print(f"   ✓ Float16 quantized: {fp16_size / 1024:.2f} KB")
    print(f"   ✓ Size reduction: {(1 - fp16_size/baseline_size) * 100:.1f}%")
    print()
    
    # 4. Export with int8 quantization (requires representative dataset)
    print("4. Exporting with int8 quantization...")
    
    def representative_dataset():
        """Generate representative data for calibration."""
        for _ in range(100):
            # Generate random data matching input shape
            yield [np.random.randn(1, 3, 224, 224).astype(np.float32)]
    
    int8_path = exporter.export_tflite(
        output_path='examples/mobile_models/model_int8.tflite',
        quantize=True,
        quantization_type='int8',
        representative_dataset=representative_dataset
    )
    int8_size = os.path.getsize(int8_path)
    print(f"   ✓ Int8 quantized: {int8_size / 1024:.2f} KB")
    print(f"   ✓ Size reduction: {(1 - int8_size/baseline_size) * 100:.1f}%")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary: Model Size Comparison")
    print("=" * 70)
    print(f"Baseline (FP32):     {baseline_size / 1024:>8.2f} KB  (100.0%)")
    print(f"Dynamic Quantized:   {dynamic_size / 1024:>8.2f} KB  ({dynamic_size/baseline_size*100:>5.1f}%)")
    print(f"Float16 Quantized:   {fp16_size / 1024:>8.2f} KB  ({fp16_size/baseline_size*100:>5.1f}%)")
    print(f"Int8 Quantized:      {int8_size / 1024:>8.2f} KB  ({int8_size/baseline_size*100:>5.1f}%)")
    print()
    print("Recommendation for mobile deployment:")
    print("  - Android/iOS (CPU): Use Int8 quantized model for best performance")
    print("  - Android/iOS (GPU): Use Float16 quantized model")
    print("  - Edge devices: Use Int8 quantized model")
    print()
    
    # Create integration examples
    create_android_example()
    create_ios_example()
    
    print("=" * 70)
    print("Mobile deployment files generated in examples/mobile_models/")
    print("=" * 70)


def create_android_example():
    """Create example code for Android integration."""
    android_code = """// Android TFLite Integration Example
// Add dependencies to build.gradle:
// implementation 'org.tensorflow:tensorflow-lite:2.12.0'

import org.tensorflow.lite.Interpreter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ModelInference {
    private Interpreter tflite;
    
    public ModelInference(String modelPath) {
        // Load model
        tflite = new Interpreter(loadModelFile(modelPath));
    }
    
    public float[] predict(float[][][][] input) {
        // Prepare input buffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
            1 * 224 * 224 * 3 * 4  // batch * height * width * channels * bytes
        );
        inputBuffer.order(ByteOrder.nativeOrder());
        
        // Fill input buffer
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                for (int c = 0; c < 3; c++) {
                    inputBuffer.putFloat(input[0][c][h][w]);
                }
            }
        }
        
        // Prepare output buffer
        float[][] output = new float[1][1000];
        
        // Run inference
        tflite.run(inputBuffer, output);
        
        return output[0];
    }
    
    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
}
"""
    
    with open('examples/mobile_models/AndroidIntegration.java', 'w') as f:
        f.write(android_code)
    
    print("   ✓ Created: examples/mobile_models/AndroidIntegration.java")


def create_ios_example():
    """Create example code for iOS integration."""
    ios_code = """// iOS TFLite Integration Example
// Add TensorFlow Lite to your Podfile:
// pod 'TensorFlowLiteSwift'

import TensorFlowLite

class ModelInference {
    private var interpreter: Interpreter?
    
    init(modelPath: String) throws {
        // Load model
        interpreter = try Interpreter(modelPath: modelPath)
        
        // Allocate tensors
        try interpreter?.allocateTensors()
    }
    
    func predict(input: [[[[Float]]]]) throws -> [Float] {
        // Prepare input data
        let inputData = Data(copyingBufferOf: input.flatMap { $0.flatMap { $0.flatMap { $0 } } })
        
        // Copy input to interpreter
        try interpreter?.copy(inputData, toInputAt: 0)
        
        // Run inference
        try interpreter?.invoke()
        
        // Get output
        let outputTensor = try interpreter?.output(at: 0)
        let outputData = outputTensor?.data
        
        // Convert output to array
        let output = [Float](unsafeData: outputData ?? Data())
        
        return output
    }
}

extension Array {
    init(unsafeData: Data) {
        self = unsafeData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Element.self))
        }
    }
}

extension Data {
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBytes { Data($0) }
    }
}
"""
    
    with open('examples/mobile_models/IOSIntegration.swift', 'w') as f:
        f.write(ios_code)
    
    print("   ✓ Created: examples/mobile_models/IOSIntegration.swift")


def create_deployment_guide():
    """Create a deployment guide for mobile."""
    guide = """# Mobile Deployment Guide

## Model Files
- `model_baseline.tflite`: Baseline model (FP32, largest)
- `model_dynamic.tflite`: Dynamic range quantization (good balance)
- `model_fp16.tflite`: Float16 quantization (GPU optimized)
- `model_int8.tflite`: Int8 quantization (smallest, fastest on CPU)

## Android Integration

1. Add TensorFlow Lite to your `build.gradle`:
   ```gradle
   dependencies {
       implementation 'org.tensorflow:tensorflow-lite:2.12.0'
   }
   ```

2. Place the model in `app/src/main/assets/`

3. Use the integration code in `AndroidIntegration.java`

## iOS Integration

1. Add TensorFlow Lite to your `Podfile`:
   ```ruby
   pod 'TensorFlowLiteSwift'
   ```

2. Place the model in your app bundle

3. Use the integration code in `IOSIntegration.swift`

## Performance Tips

### Android
- Use NNAPI delegate for hardware acceleration:
  ```java
  Interpreter.Options options = new Interpreter.Options();
  options.addDelegate(new NnApiDelegate());
  tflite = new Interpreter(modelFile, options);
  ```

### iOS
- Use Metal delegate for GPU acceleration:
  ```swift
  var options = Interpreter.Options()
  options.threadCount = 2
  let metalDelegate = MetalDelegate()
  options.delegates = [metalDelegate]
  ```

## Testing

Test on actual devices with various:
- Screen sizes
- OS versions
- Hardware capabilities (CPU, GPU, NPU)

## Monitoring

Track:
- Inference latency
- Memory usage
- Battery consumption
- Model accuracy
"""
    
    with open('examples/mobile_models/README.md', 'w') as f:
        f.write(guide)
    
    print("   ✓ Created: examples/mobile_models/README.md")


def main():
    """Run the mobile deployment example."""
    try:
        export_for_mobile()
        create_deployment_guide()
        print()
        print("✓ Mobile deployment example completed successfully!")
        print()
        print("Next steps:")
        print("  1. Review the generated models in examples/mobile_models/")
        print("  2. Choose the appropriate model for your target device")
        print("  3. Follow the integration guides for Android or iOS")
        print("  4. Test on actual devices")
        print()
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
