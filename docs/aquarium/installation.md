# Aquarium IDE Installation Guide

## Overview

Aquarium IDE is a modern, web-based Integrated Development Environment for Neural DSL. This guide covers installation, setup, and troubleshooting for all supported platforms.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+, Fedora 30+)
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Storage**: 500 MB for Aquarium + dependencies
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+

### Recommended Requirements
- **RAM**: 16 GB or more (for large models)
- **GPU**: CUDA-compatible GPU for TensorFlow/PyTorch acceleration
- **Storage**: 5+ GB for backends and datasets
- **Display**: 1920x1080 or higher resolution

## Installation Methods

### Method 1: Install with Neural DSL (Recommended)

If you already have Neural DSL installed:

```bash
# Install Neural DSL with dashboard support
pip install neural-dsl[dashboard]

# Or install full package (includes all features)
pip install neural-dsl[full]
```

### Method 2: Install from Source

For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate

# macOS/Linux
source .venv/bin/activate

# Install with dashboard dependencies
pip install -e ".[dashboard]"

# Or install all features
pip install -e ".[full]"
```

### Method 3: Minimal Installation

For core Aquarium functionality only:

```bash
# Install core Neural DSL
pip install neural-dsl

# Install Aquarium dependencies manually
pip install dash dash-bootstrap-components plotly
```

## Backend Installation

Aquarium supports multiple ML backends. Install the ones you need:

### TensorFlow Backend

```bash
# CPU version (lightweight)
pip install tensorflow-cpu

# GPU version (requires CUDA)
pip install tensorflow

# Or via Neural DSL extras
pip install neural-dsl[backends]
```

### PyTorch Backend

```bash
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU version (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### ONNX Backend

```bash
pip install onnx onnxruntime
```

## Verification

### Verify Installation

```bash
# Check Neural DSL installation
python -c "import neural; print(f'Neural DSL version: {neural.__version__}')"

# Check Aquarium availability
python -c "from neural.aquarium import aquarium; print('Aquarium: OK')"

# Check Dash installation
python -c "import dash; print(f'Dash version: {dash.__version__}')"
```

### Launch Aquarium

```bash
# Start Aquarium IDE
python -m neural.aquarium.aquarium

# Or with custom port
python -m neural.aquarium.aquarium --port 8052
```

Expected output:
```
======================================================================
    _   __                      __    ___                            _                 
   / | / /__  __  ___________  / /   /   | ____ ___  ______ ______(_)_  ______ ___   
  /  |/ / _ \/ / / / ___/ __ \/ /   / /| |/ __ `/ / / / __ `/ ___/ / / / / __ `__ \  
 / /|  /  __/ /_/ / /  / /_/ / /   / ___ / /_/ / /_/ / /_/ / /  / / /_/ / / / / / /  
/_/ |_/\___/\__,_/_/   \____/_/   /_/  |_\__, /\__,_/\__,_/_/  /_/\__,_/_/ /_/ /_/   
                                            /_/                                        
======================================================================

🚀 Starting Neural Aquarium IDE on http://localhost:8052
   Backend: Dash + Plotly
   Debug Mode: False

📝 Features:
   • DSL Editor with syntax validation
   • Model Compilation (TensorFlow, PyTorch, ONNX)
   • Execution Panel with live logs
   • Dataset Selection (MNIST, CIFAR10, CIFAR100, ImageNet)
   • Export and IDE Integration

🌐 Open your browser to: http://localhost:8052
   Press Ctrl+C to stop the server

======================================================================
```

### Access the IDE

Open your browser and navigate to:
```
http://localhost:8052
```

## Platform-Specific Setup

### Windows Setup

```powershell
# Install Python from python.org (if not installed)
# Or use Microsoft Store

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium
```

**Windows Firewall**: Allow Python through the firewall when prompted.

### macOS Setup

```bash
# Install Python via Homebrew (recommended)
brew install python@3.11

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium
```

**macOS Security**: Allow network connections when prompted.

### Linux Setup

#### Ubuntu/Debian

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium
```

#### Fedora/RHEL

```bash
# Install Python
sudo dnf install python3 python3-pip

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium
```

#### Arch Linux

```bash
# Install Python
sudo pacman -S python python-pip

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium
```

## Docker Installation

### Using Docker

```bash
# Build Docker image
docker build -t neural-aquarium .

# Run container
docker run -p 8052:8052 neural-aquarium

# Or with volume mount for persistence
docker run -p 8052:8052 -v $(pwd)/models:/app/models neural-aquarium
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  aquarium:
    build: .
    ports:
      - "8052:8052"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## GPU Support

### NVIDIA GPU (CUDA)

#### Windows/Linux

```bash
# Install CUDA toolkit (version 11.8 or 12.1)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# Download from: https://developer.nvidia.com/cudnn

# Install TensorFlow GPU
pip install tensorflow[and-cuda]

# Or PyTorch GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### AMD GPU (ROCm) - Linux Only

```bash
# Install ROCm
# Follow instructions at: https://rocm.docs.amd.com/

# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

### Apple Silicon (Metal)

```bash
# TensorFlow Metal plugin (macOS only)
pip install tensorflow-metal

# PyTorch with MPS support (included by default)
pip install torch torchvision
```

## Configuration

### Configuration File

Create `~/.neural/aquarium/config.yaml`:

```yaml
# Aquarium Configuration
server:
  host: localhost
  port: 8052
  debug: false

paths:
  compiled: ~/.neural/aquarium/compiled
  exported: ~/.neural/aquarium/exported
  temp: ~/.neural/aquarium/temp

backends:
  default: tensorflow
  available:
    - tensorflow
    - pytorch
    - onnx

datasets:
  cache_dir: ~/.neural/datasets
  auto_download: true

training:
  default_epochs: 10
  default_batch_size: 32
  default_validation_split: 0.2

ui:
  theme: darkly
  syntax_highlighting: true
  auto_save: true
```

### Environment Variables

```bash
# Set environment variables (optional)
export NEURAL_AQUARIUM_PORT=8052
export NEURAL_AQUARIUM_HOST=localhost
export NEURAL_CACHE_DIR=~/.neural/cache
export NEURAL_LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

#### Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Use a different port
python -m neural.aquarium.aquarium --port 8053

# Or find and kill the process using the port
# Windows
netstat -ano | findstr :8052
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:8052 | xargs kill -9
```

#### Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'dash'`

**Solution**:
```bash
# Install missing dependencies
pip install dash dash-bootstrap-components plotly

# Or reinstall with full dependencies
pip install --force-reinstall neural-dsl[dashboard]
```

#### Browser Can't Connect

**Problem**: Browser shows "Can't reach this page"

**Solution**:
1. Verify server is running (check terminal output)
2. Check firewall settings
3. Try accessing via `http://127.0.0.1:8052` instead
4. Check if another application is using the port

#### Import Errors

**Problem**: `ImportError: cannot import name 'aquarium'`

**Solution**:
```bash
# Ensure Neural DSL is properly installed
pip uninstall neural-dsl
pip install neural-dsl[dashboard]

# Or reinstall from source
cd Neural
pip install -e ".[dashboard]"
```

#### Performance Issues

**Problem**: Slow UI or laggy interface

**Solution**:
1. Close other browser tabs
2. Reduce batch size in training config
3. Use a modern browser (Chrome/Firefox recommended)
4. Increase RAM allocation
5. Check for background processes consuming resources

### Platform-Specific Issues

#### Windows: Python Not Recognized

**Solution**:
```powershell
# Add Python to PATH
# During installation, check "Add Python to PATH"
# Or manually add: C:\Python3X and C:\Python3X\Scripts

# Verify
python --version
```

#### macOS: SSL Certificate Errors

**Solution**:
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or use Homebrew Python
brew install python@3.11
```

#### Linux: Permission Denied

**Solution**:
```bash
# Fix permissions for virtual environment
chmod -R u+w .venv

# Or install with user flag
pip install --user neural-dsl[dashboard]
```

## Upgrading

### Upgrade to Latest Version

```bash
# Upgrade Neural DSL and Aquarium
pip install --upgrade neural-dsl[dashboard]

# Verify new version
python -c "import neural; print(neural.__version__)"
```

### Upgrade from Source

```bash
cd Neural
git pull origin main
pip install --upgrade -e ".[dashboard]"
```

## Uninstallation

### Remove Aquarium

```bash
# Uninstall Neural DSL (includes Aquarium)
pip uninstall neural-dsl

# Remove configuration and cache
rm -rf ~/.neural/aquarium
rm -rf ~/.neural/cache

# Remove virtual environment (if used)
rm -rf .venv
```

## Next Steps

After successful installation:

1. **Read the User Manual**: [user-manual.md](user-manual.md)
2. **Try the Quick Start**: Launch Aquarium and load an example model
3. **Explore Features**: Check [architecture.md](architecture.md) for detailed features
4. **Join the Community**: Visit [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)

## Getting Help

- **Documentation**: [README.md](../../neural/aquarium/README.md)
- **GitHub Issues**: [Report bugs](https://github.com/Lemniscate-world/Neural/issues)
- **Discord**: [Join community](https://discord.gg/KFku4KvS)
- **Email**: Lemniscate_zero@proton.me

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
