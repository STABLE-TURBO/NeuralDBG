# Neural No-Code Interface

<p align="center">
  <img src="../../docs/images/no_code_interface.png" alt="No-Code Interface" width="800"/>
</p>

The Neural No-Code Interface is a graphical user interface for building, visualizing, and debugging neural networks without writing code. It provides a user-friendly way to create models using Neural DSL and export them to various backends.

## Features

- **Intuitive Model Building**: Drag-and-drop interface for creating neural network architectures
- **Layer Configuration**: Configure layer parameters with a user-friendly interface
- **Shape Propagation**: Visualize tensor shapes as they flow through the network
- **Code Generation**: Generate Neural DSL, TensorFlow, and PyTorch code
- **Model Management**: Save and load models for later use
- **NeuralDbg Integration**: Launch the NeuralDbg dashboard for real-time debugging
- **Dark Theme**: Modern, eye-friendly dark interface

## Getting Started

### 1. Launch the No-Code Interface

```bash
neural no-code
```

This will start the interface on http://localhost:8051.

### 2. Create a Model

1. **Select an Input Shape**: Choose from common input shapes (MNIST, CIFAR-10, ImageNet) or define a custom shape
2. **Add Layers**: Select layer types from the sidebar and configure their parameters
3. **Visualize**: View the model architecture and shape propagation in the Visualization tab
4. **Generate Code**: Generate Neural DSL, TensorFlow, and PyTorch code in the Generated Code tab

### 3. Save and Load Models

- **Save Model**: Click the "Save Model" button to save your model for later use
- **Load Model**: Click the "Load Model" button to load a previously saved model
- **Export DSL**: Click the "Export DSL" button to export your model as a Neural DSL file

### 4. Debug with NeuralDbg

1. Click the "Debug" tab
2. Click the "Launch NeuralDbg" button
3. The NeuralDbg dashboard will open in a new tab, allowing you to debug your model in real-time

## Model Templates

The No-Code Interface includes several pre-defined model templates:

- **Simple CNN for MNIST**: A basic convolutional neural network for MNIST digit classification
- **VGG-like for CIFAR-10**: A VGG-style network for CIFAR-10 image classification
- **Simple LSTM for Text**: An LSTM-based network for text classification
- **Transformer Encoder**: A transformer encoder for sequence processing

## Layer Types

The No-Code Interface supports a wide range of layer types:

- **Convolutional**: Conv1D, Conv2D, Conv3D, SeparableConv2D, DepthwiseConv2D, TransposedConv2D
- **Pooling**: MaxPooling1D/2D, AveragePooling1D/2D, GlobalMaxPooling1D/2D, GlobalAveragePooling1D/2D
- **Core**: Dense, Flatten, Reshape, Permute, RepeatVector, Lambda
- **Normalization**: BatchNormalization, LayerNormalization, GroupNormalization
- **Regularization**: Dropout, SpatialDropout1D/2D, GaussianNoise, GaussianDropout, ActivityRegularization
- **Recurrent**: LSTM, GRU, SimpleRNN, Bidirectional, ConvLSTM2D
- **Attention**: MultiHeadAttention, Attention
- **Embedding**: Embedding
- **Activation**: ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, Sigmoid, Tanh
- **Output**: Output

## Implementation Details

The No-Code Interface is built using:

- **Dash**: A Python framework for building web applications
- **Plotly**: For interactive visualizations
- **Bootstrap**: For responsive UI components
- **Neural DSL**: For model definition and code generation

The interface is designed to be modular and extensible, making it easy to add new features and layer types.

## Future Enhancements

- **Drag-and-Drop Layer Ordering**: Reorder layers by dragging and dropping
- **Custom Layer Support**: Add support for custom layers
- **Training Interface**: Train models directly from the interface
- **Export to ONNX**: Add support for exporting to ONNX format
- **Interactive Tutorials**: Add interactive tutorials for common tasks
