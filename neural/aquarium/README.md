# Neural Aquarium IDE

A modern, web-based Integrated Development Environment (IDE) for Neural DSL.

## Features

### 🎨 DSL Editor
- Syntax-highlighted editor for Neural DSL
- Real-time validation
- Example models
- Parse and validate DSL code

### 🔧 Model Compilation & Execution Panel
- **Backend Selection**: Choose between TensorFlow, PyTorch, or ONNX
- **Dataset Selection**: Built-in support for MNIST, CIFAR10, CIFAR100, ImageNet, or custom datasets
- **Training Configuration**:
  - Adjustable epochs, batch size, and validation split
  - Auto-flatten output option
  - HPO (Hyperparameter Optimization) support
  - Verbose logging
  - Model weight saving

### 🏃 Execution Features
- **Compile**: Generate backend-specific Python code from Neural DSL
- **Run**: Execute training scripts directly from the IDE
- **Stop**: Terminate running processes
- **Live Console**: Real-time compilation and training logs
- **Training Metrics**: Visualize loss and accuracy during training

### 📦 Export & Integration
- **Export Script**: Save generated Python scripts to custom locations
- **Open in IDE**: Launch scripts in your default code editor
- **Script Management**: Organize and manage compiled scripts

### 🐛 Debugging
- Integration with NeuralDbg for advanced debugging
- Real-time model visualization
- Layer-by-layer inspection

## Quick Start

### Running Aquarium

```bash
# From the repository root
python -m neural.aquarium.aquarium

# Or with custom port
python -m neural.aquarium.aquarium --port 8052

# Debug mode
python -m neural.aquarium.aquarium --debug
```

### Using the IDE

1. **Write DSL Code**: Use the editor to write your Neural DSL model
   ```neural
   network MyModel {
       input: (None, 28, 28, 1)
       layers:
           Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
           MaxPooling2D(pool_size=(2, 2))
           Flatten()
           Dense(units=128, activation=relu)
           Output(units=10, activation=softmax)
       loss: categorical_crossentropy
       optimizer: Adam(learning_rate=0.001)
   }
   ```

2. **Parse DSL**: Click "Parse DSL" to validate your model

3. **Configure Execution**:
   - Select backend (TensorFlow/PyTorch/ONNX)
   - Choose dataset (MNIST, CIFAR10, etc.)
   - Set training parameters (epochs, batch size, validation split)

4. **Compile**: Click "Compile" to generate Python code

5. **Run**: Click "Run" to execute the training script

6. **Monitor**: Watch real-time logs and training metrics

7. **Export**: Save the generated script for later use or modification

## Architecture

```
neural/aquarium/
├── aquarium.py              # Main application entry point
├── __init__.py              # Package initialization
├── README.md                # This file
└── src/
    └── components/
        └── runner/
            ├── __init__.py            # Runner panel exports
            ├── runner_panel.py        # Main UI panel component
            ├── execution_manager.py   # Process and execution management
            └── script_generator.py    # Training script generation
```

## Components

### Runner Panel (`runner_panel.py`)
The main UI component providing:
- Backend selection dropdown
- Dataset configuration
- Training parameters
- Action buttons (Compile, Run, Stop, Export, Open in IDE, Clear)
- Console output display
- Training metrics visualization

### Execution Manager (`execution_manager.py`)
Handles:
- Model compilation
- Script execution in separate processes
- Output stream capture
- Metrics parsing
- Process lifecycle management

### Script Generator (`script_generator.py`)
Generates complete training scripts:
- Dataset loading code
- Model building code
- Training loop
- Evaluation code
- Model saving logic

## Usage Examples

### Example 1: Training MNIST Classifier

1. Load the MNIST example
2. Select "TensorFlow" backend
3. Select "MNIST" dataset
4. Set epochs to 10
5. Click "Compile" then "Run"

### Example 2: Exporting a Script

1. After compilation, click "Export Script"
2. Enter a filename (e.g., "my_model.py")
3. Specify export location
4. Click "Export"

### Example 3: Custom Dataset

1. Select "Custom" from dataset dropdown
2. Enter path to your dataset directory
3. Configure training parameters
4. Compile and run

## Integration with Neural CLI

Aquarium can be launched from the Neural CLI:

```bash
# Future integration (to be implemented)
neural aquarium
neural aquarium --port 8052
```

## Requirements

- Python 3.8+
- Dash
- Dash Bootstrap Components
- Neural DSL (core package)
- TensorFlow (for TensorFlow backend)
- PyTorch (for PyTorch backend)
- ONNX (for ONNX backend)

## Tips

- Use the "Parse DSL" button frequently to catch syntax errors early
- Monitor the console output for compilation and training progress
- Use "Stop" button to terminate long-running training sessions
- Export scripts for fine-tuning or production deployment
- Enable HPO for automated hyperparameter optimization

## Troubleshooting

### Port Already in Use
```bash
python -m neural.aquarium.aquarium --port 8053
```

### Missing Dependencies
```bash
pip install -e ".[full]"  # Install all optional dependencies
```

### Script Execution Fails
- Check that the selected dataset matches your model's input shape
- Verify that the backend is properly installed
- Review console logs for specific error messages

## Future Enhancements

- [ ] Syntax highlighting in DSL editor
- [ ] Code completion and IntelliSense
- [ ] Model comparison tools
- [ ] Experiment tracking integration
- [ ] Cloud execution support
- [ ] Collaborative editing
- [ ] Version control integration

## Contributing

To contribute to Aquarium:
1. Follow the coding style in existing components
2. Add tests for new features
3. Update documentation
4. Submit a pull request

## License

Same as Neural DSL package (see repository LICENSE file)
