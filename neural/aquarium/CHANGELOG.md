# Neural Aquarium Changelog

All notable changes to Neural Aquarium will be documented in this file.

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Neural Aquarium IDE
- DSL Editor with validation
- Model compilation panel supporting TensorFlow, PyTorch, and ONNX backends
- Dataset selection interface (MNIST, CIFAR10, CIFAR100, ImageNet, Custom)
- Execution panel with real-time console output
- Training configuration (epochs, batch size, validation split)
- Training options (auto-flatten, HPO, verbose, save weights)
- Live execution logs with process management
- Compile, Run, Stop, Export, Open in IDE, and Clear actions
- Export modal for saving generated scripts
- Integration with Neural CLI compile command
- Script generation for complete training pipelines
- Process lifecycle management (start, stop, monitor)
- Metrics parsing and visualization support
- Multi-tab interface (Runner, Debugger, Visualization, Documentation)
- Example model loader
- Model information display
- Dark theme UI with Bootstrap styling

### Features

#### Runner Panel
- Backend selection dropdown (TensorFlow/PyTorch/ONNX)
- Dataset selection with custom path support
- Training parameter configuration
- Action buttons for model lifecycle
- Real-time console output with color coding
- Metrics visualization placeholder
- Modal dialogs for export configuration

#### Execution Manager
- Model compilation using Neural code generator
- Script execution in separate processes
- Output stream capture and queuing
- Metrics parsing from training logs
- Process termination support
- Script export functionality
- IDE integration for opening scripts

#### Script Generator
- Complete training script generation
- Dataset loading code
- Model building wrappers
- Training loop implementation
- Evaluation logic
- Model weight saving
- Backend-specific templates (TensorFlow/PyTorch)

### Technical Details
- Built with Dash and Dash Bootstrap Components
- Async process execution with threading
- Queue-based log streaming
- Modular component architecture
- Type hints for better code quality

### Documentation
- Comprehensive README with architecture overview
- Quick Start guide for new users
- Usage examples and common tasks
- Troubleshooting section
- API documentation in docstrings

## [Unreleased]

### Planned Features
- Syntax highlighting in DSL editor
- Code completion and IntelliSense
- Real-time metrics visualization during training
- Model comparison tools
- Experiment tracking integration
- HPO integration in UI
- Cloud execution support
- Collaborative editing
- Version control integration
- Custom theme support
- Keyboard shortcuts
- Model templates library
- Auto-save functionality
- Search and replace in editor
- Multi-file project support

### Known Issues
- Metrics graph not yet connected to live data
- NeuralDbg integration placeholder
- Architecture visualization basic implementation
- No syntax highlighting in editor yet

## Contributing

See CONTRIBUTING.md for guidelines on contributing to Neural Aquarium.
