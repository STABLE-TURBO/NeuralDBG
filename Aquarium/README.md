# Aquarium: Neural Network IDE

Aquarium is a specialized IDE for designing, training, debugging, and deploying neural networks using the Neural framework. It provides a visual interface for neural network development with real-time shape propagation and error detection.

## Features

- **Visual Network Designer**: Drag-and-drop interface for creating neural network architectures
- **Real-time Shape Propagation**: Visualize tensor dimensions as they flow through your network
- **Integrated Debugging**: Catch dimensional errors before training begins
- **Neural DSL Integration**: Automatically generate Neural DSL code from visual designs
- **Performance Analysis**: Estimate computational requirements and memory usage

## Technology Stack

- **Frontend**: Tauri with JavaScript/HTML/CSS
- **Backend**: Rust for performance-critical components
- **Neural Integration**: Direct integration with Neural's shape propagator and other components

## Development Setup

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [Node.js](https://nodejs.org/)
- [Tauri CLI](https://tauri.app/v1/guides/getting-started/prerequisites)

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aquarium.git
   cd aquarium
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   cargo tauri dev
   ```

## Building for Production

To build the application for production:

```bash
cargo tauri build
```

## Integration with Neural

Aquarium integrates with the Neural framework to provide:

1. **Shape Propagation**: Leverages Neural's shape propagator to calculate tensor dimensions
2. **Code Generation**: Generates Neural DSL code from visual designs
3. **Training Integration**: Connects with Neural's training capabilities
4. **Debugging Tools**: Provides insights into model performance and potential issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) with:
  - [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode)
  - [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
