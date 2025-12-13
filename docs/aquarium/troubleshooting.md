# Aquarium IDE Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Launch Problems](#launch-problems)
3. [Connection Issues](#connection-issues)
4. [Compilation Errors](#compilation-errors)
5. [Execution Problems](#execution-problems)
6. [Performance Issues](#performance-issues)
7. [UI/Display Issues](#uidisplay-issues)
8. [Export/Integration Issues](#exportintegration-issues)
9. [Platform-Specific Issues](#platform-specific-issues)
10. [Getting Help](#getting-help)

## Installation Issues

### Problem: pip install fails

**Symptom**:
```
ERROR: Could not find a version that satisfies the requirement neural-dsl
```

**Solutions**:

1. **Update pip**:
```bash
python -m pip install --upgrade pip
```

2. **Check Python version**:
```bash
python --version  # Must be 3.8+
```

3. **Try with extras**:
```bash
pip install neural-dsl[dashboard]
```

4. **Install from source**:
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[dashboard]"
```

---

### Problem: Missing dependencies

**Symptom**:
```
ModuleNotFoundError: No module named 'dash'
```

**Solutions**:

1. **Install dashboard extras**:
```bash
pip install neural-dsl[dashboard]
```

2. **Manual installation**:
```bash
pip install dash dash-bootstrap-components plotly
```

3. **Verify installation**:
```bash
python -c "import dash; print(dash.__version__)"
```

---

### Problem: Virtual environment issues

**Symptom**:
- Can't activate virtual environment
- Packages not found after installation

**Solutions**:

**Windows**:
```powershell
# Create new venv
python -m venv .venv
.\.venv\Scripts\Activate

# If execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux**:
```bash
# Create new venv
python3 -m venv .venv
source .venv/bin/activate

# If permission issues
chmod +x .venv/bin/activate
```

---

### Problem: Conflicting dependencies

**Symptom**:
```
ERROR: neural-dsl has requirement dash>=2.0.0, but you have dash 1.x.x
```

**Solutions**:

1. **Create fresh environment**:
```bash
python -m venv fresh_venv
source fresh_venv/bin/activate  # or .venv\Scripts\Activate on Windows
pip install neural-dsl[dashboard]
```

2. **Force reinstall**:
```bash
pip install --force-reinstall neural-dsl[dashboard]
```

## Launch Problems

### Problem: Port already in use

**Symptom**:
```
OSError: [Errno 48] Address already in use
```

**Solutions**:

1. **Use different port**:
```bash
python -m neural.aquarium.aquarium --port 8053
```

2. **Find and kill process** (Windows):
```powershell
netstat -ano | findstr :8052
taskkill /PID <PID> /F
```

3. **Find and kill process** (macOS/Linux):
```bash
lsof -ti:8052 | xargs kill -9
```

---

### Problem: Module not found on launch

**Symptom**:
```
ModuleNotFoundError: No module named 'neural.aquarium'
```

**Solutions**:

1. **Verify installation**:
```bash
python -c "import neural.aquarium; print('OK')"
```

2. **Check PYTHONPATH**:
```bash
# Add to PYTHONPATH
export PYTHONPATH=/path/to/Neural:$PYTHONPATH  # Linux/macOS
set PYTHONPATH=C:\path\to\Neural;%PYTHONPATH%  # Windows
```

3. **Install in editable mode**:
```bash
cd Neural
pip install -e .
```

---

### Problem: Permission denied

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '/home/user/.neural'
```

**Solutions**:

1. **Fix permissions**:
```bash
chmod -R u+w ~/.neural
```

2. **Run without sudo**:
```bash
# Don't use sudo for Python packages
pip install --user neural-dsl[dashboard]
```

## Connection Issues

### Problem: Browser can't connect

**Symptom**:
- "This site can't be reached"
- Connection refused

**Solutions**:

1. **Verify server is running**:
   - Check terminal for "Running on http://..."
   - Look for error messages

2. **Try different address**:
   - `http://localhost:8052`
   - `http://127.0.0.1:8052`
   - `http://0.0.0.0:8052`

3. **Check firewall**:
```bash
# Windows: Allow Python through firewall
# Linux: Allow port 8052
sudo ufw allow 8052

# macOS: System Preferences → Security & Privacy → Firewall
```

4. **Restart server**:
```bash
# Stop with Ctrl+C
# Restart
python -m neural.aquarium.aquarium
```

---

### Problem: Slow or unresponsive UI

**Symptom**:
- UI takes long to load
- Buttons don't respond
- Page freezes

**Solutions**:

1. **Clear browser cache**:
   - Chrome: Ctrl+Shift+Delete
   - Firefox: Ctrl+Shift+Del
   - Safari: Cmd+Option+E

2. **Try different browser**:
   - Chrome (recommended)
   - Firefox
   - Edge

3. **Check network**:
```bash
ping localhost
```

4. **Restart in debug mode**:
```bash
python -m neural.aquarium.aquarium --debug
```

---

### Problem: WebSocket errors

**Symptom**:
```
WebSocket connection failed
```

**Solutions**:

1. **Disable browser extensions**:
   - Ad blockers can interfere
   - Try incognito/private mode

2. **Check proxy settings**:
   - Disable VPN temporarily
   - Check system proxy settings

3. **Use hard refresh**:
   - Chrome/Firefox: Ctrl+Shift+R
   - Safari: Cmd+Option+R

## Compilation Errors

### Problem: Parse error

**Symptom**:
```
Parse error: Expected 'network' at line X
```

**Solutions**:

1. **Check DSL syntax**:
```neural
# Correct syntax
network MyModel {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

2. **Common mistakes**:
   - Missing colons after `input`, `layers`, etc.
   - Incorrect indentation
   - Missing commas in tuples
   - Typos in layer names

3. **Use Load Example**:
   - Start with working example
   - Modify incrementally

---

### Problem: Backend not supported

**Symptom**:
```
Backend 'xyz' not recognized
```

**Solutions**:

1. **Check available backends**:
   - TensorFlow
   - PyTorch
   - ONNX

2. **Install backend**:
```bash
# TensorFlow
pip install tensorflow

# PyTorch
pip install torch torchvision

# ONNX
pip install onnx onnxruntime
```

3. **Verify installation**:
```bash
python -c "import tensorflow; print('TF OK')"
python -c "import torch; print('PyTorch OK')"
```

---

### Problem: Code generation fails

**Symptom**:
```
[ERROR] Failed to generate code for layer X
```

**Solutions**:

1. **Check layer compatibility**:
   - Some layers not supported in all backends
   - Check documentation for backend support

2. **Simplify model**:
   - Remove advanced layers
   - Use basic layers first

3. **Update Neural DSL**:
```bash
pip install --upgrade neural-dsl
```

## Execution Problems

### Problem: Script won't run

**Symptom**:
- "Run" button does nothing
- Process starts then immediately stops

**Solutions**:

1. **Compile first**:
   - Click "Parse DSL"
   - Click "Compile"
   - Then click "Run"

2. **Check console output**:
   - Look for error messages
   - Check for missing dependencies

3. **Check dataset**:
   - Verify dataset is selected
   - Ensure dataset matches input shape

---

### Problem: Import errors during execution

**Symptom**:
```
[RUN ERROR] ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions**:

1. **Install missing backend**:
```bash
pip install tensorflow  # or pytorch, onnx
```

2. **Verify backend installation**:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

3. **Check virtual environment**:
   - Ensure correct venv is activated
   - Reinstall packages if needed

---

### Problem: Dataset loading fails

**Symptom**:
```
[RUN ERROR] Failed to load dataset: MNIST
```

**Solutions**:

1. **First-time download**:
   - Datasets download on first use
   - Be patient, may take several minutes
   - Check internet connection

2. **Manual dataset setup**:
```python
# For MNIST
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

3. **Check cache directory**:
```bash
# Clear corrupted cache
rm -rf ~/.keras/datasets  # TensorFlow
rm -rf ~/.cache/torch     # PyTorch
```

---

### Problem: Out of memory

**Symptom**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:

1. **Reduce batch size**:
   - Try 16, 8, or even 4
   - Smaller = slower but less memory

2. **Reduce model size**:
   - Fewer layers
   - Fewer filters/units
   - Smaller input size

3. **Enable GPU memory growth** (TensorFlow):
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

4. **Close other applications**:
   - Free up RAM
   - Close browser tabs
   - Stop unnecessary processes

---

### Problem: Training is very slow

**Symptom**:
- Each epoch takes many minutes
- Progress bar barely moves

**Solutions**:

1. **Check GPU usage**:
```bash
# NVIDIA GPU
nvidia-smi

# Check GPU is being used
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

2. **Reduce dataset size** (for testing):
   - Use fewer epochs (5 instead of 100)
   - Smaller batch size
   - Subset of data

3. **Optimize data loading**:
   - Enable data preprocessing
   - Use data augmentation wisely
   - Check disk I/O

---

### Problem: Can't stop execution

**Symptom**:
- "Stop" button doesn't work
- Process keeps running

**Solutions**:

1. **Click Stop again**:
   - May take a moment to respond
   - Be patient

2. **Force kill process** (Windows):
```powershell
tasklist | findstr python
taskkill /F /PID <PID>
```

3. **Force kill process** (macOS/Linux):
```bash
ps aux | grep python
kill -9 <PID>
```

4. **Restart Aquarium**:
   - Close browser tab
   - Stop server (Ctrl+C)
   - Restart

## Performance Issues

### Problem: UI is laggy

**Solutions**:

1. **Clear console**:
   - Click "Clear" button
   - Reduces DOM size

2. **Reduce console buffer**:
   - Edit `execution_manager.py`
   - Reduce max output lines

3. **Close unused tabs**:
   - Switch to Runner tab only
   - Close other browser tabs

4. **Increase browser memory**:
   - Close other applications
   - Restart browser

---

### Problem: High CPU usage

**Solutions**:

1. **Normal during training**:
   - High CPU is expected
   - Training is CPU/GPU intensive

2. **Idle high CPU**:
   - Check for background processes
   - Restart Aquarium
   - Check for infinite loops

3. **Monitor resources**:
```bash
# Windows
taskmgr

# macOS
Activity Monitor

# Linux
htop
```

## UI/Display Issues

### Problem: Layout is broken

**Symptom**:
- Overlapping elements
- Missing components
- Misaligned text

**Solutions**:

1. **Hard refresh**:
   - Ctrl+Shift+R (Chrome/Firefox)
   - Cmd+Shift+R (Safari)

2. **Clear browser cache**:
   - Settings → Privacy → Clear browsing data

3. **Try different browser**:
   - Chrome (recommended)
   - Firefox
   - Edge

4. **Check zoom level**:
   - Reset to 100% (Ctrl+0)

---

### Problem: Icons not showing

**Symptom**:
- Boxes instead of icons
- Missing Font Awesome icons

**Solutions**:

1. **Check internet connection**:
   - Icons load from CDN
   - Requires active connection

2. **Disable ad blocker**:
   - May block icon CDN
   - Try incognito mode

3. **Clear cache**:
   - Hard refresh page

---

### Problem: Dark theme not applied

**Solutions**:

1. **Verify theme setting**:
   - Check `config.py`
   - Ensure `theme: darkly`

2. **Clear browser cache**

3. **Restart server**:
```bash
python -m neural.aquarium.aquarium
```

## Export/Integration Issues

### Problem: Export fails

**Symptom**:
```
Failed to export script
```

**Solutions**:

1. **Check permissions**:
```bash
# Ensure write access
chmod +w ~/path/to/export/
```

2. **Verify path exists**:
```bash
mkdir -p ~/path/to/export/
```

3. **Check disk space**:
```bash
df -h  # Linux/macOS
```

---

### Problem: Can't open in IDE

**Symptom**:
- "Open in IDE" does nothing
- Error message appears

**Solutions**:

1. **Set default program** (Windows):
   - Right-click .py file
   - Open with → Choose default program
   - Select Python IDE (VS Code, PyCharm, etc.)

2. **macOS**:
```bash
# Set default app
defaults write com.apple.LaunchServices/com.apple.launchservices.secure LSHandlers -array-add '{LSHandlerContentType=public.python-script;LSHandlerRoleAll=com.microsoft.VSCode;}'
```

3. **Linux**:
```bash
# Set default with xdg
xdg-mime default code.desktop text/x-python
```

4. **Manual open**:
   - Navigate to exported file
   - Open manually in IDE

## Platform-Specific Issues

### Windows Issues

#### Problem: Python not in PATH

**Solution**:
1. Reinstall Python, check "Add Python to PATH"
2. Or manually add:
   - System → Advanced → Environment Variables
   - Edit PATH, add `C:\Python3X` and `C:\Python3X\Scripts`

#### Problem: Long path errors

**Solution**:
1. Enable long paths:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Or use shorter paths

### macOS Issues

#### Problem: SSL certificate errors

**Solution**:
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or use Homebrew Python
brew install python@3.11
```

#### Problem: Permission denied for port

**Solution**:
```bash
# Use port >1024 (doesn't require sudo)
python -m neural.aquarium.aquarium --port 8052
```

### Linux Issues

#### Problem: Display not set

**Solution**:
```bash
export DISPLAY=:0
```

#### Problem: Missing system libraries

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install python3-dev libffi-dev

# Fedora
sudo dnf install python3-devel libffi-devel
```

## Debug Mode

Enable debug mode for detailed error messages:

```bash
python -m neural.aquarium.aquarium --debug
```

**Debug features**:
- Detailed error traces
- More verbose logging
- Hot reload on code changes
- Flask debug toolbar

## Logging

### Enable logging

```bash
export NEURAL_LOG_LEVEL=DEBUG  # Linux/macOS
set NEURAL_LOG_LEVEL=DEBUG     # Windows

python -m neural.aquarium.aquarium
```

### Log locations

- **stdout**: Terminal output
- **File**: `~/.neural/aquarium/logs/aquarium.log`

### Check logs

```bash
# View logs
tail -f ~/.neural/aquarium/logs/aquarium.log

# Windows
type %USERPROFILE%\.neural\aquarium\logs\aquarium.log
```

## Getting Help

### Before asking for help

1. **Check this guide** for your issue
2. **Search GitHub Issues**: [Issues](https://github.com/Lemniscate-world/Neural/issues)
3. **Check documentation**: [README](../../neural/aquarium/README.md)
4. **Try debug mode**: `--debug` flag
5. **Collect error messages** and logs

### How to ask for help

**Good bug report includes**:
- Aquarium version: `python -c "import neural; print(neural.__version__)"`
- Python version: `python --version`
- Operating system: Windows/macOS/Linux
- Exact error message
- Steps to reproduce
- What you've tried

**Example**:
```
**Environment:**
- Aquarium: 0.1.0
- Python: 3.9.7
- OS: Windows 11

**Issue:**
Export fails with permission error

**Error:**
PermissionError: [Errno 13] Permission denied: 'C:\\exports\\model.py'

**Steps:**
1. Compile model
2. Click Export
3. Enter filename: model.py
4. Select location: C:\exports\
5. Click Export button

**What I tried:**
- Checked folder permissions
- Tried different location
- Restarted Aquarium
```

### Where to get help

1. **GitHub Discussions**: [Discussions](https://github.com/Lemniscate-world/Neural/discussions)
   - Q&A, feature requests
   - Community support

2. **GitHub Issues**: [Issues](https://github.com/Lemniscate-world/Neural/issues)
   - Bug reports
   - Technical issues

3. **Discord**: [Join Discord](https://discord.gg/KFku4KvS)
   - Real-time chat
   - Community help

4. **Email**: Lemniscate_zero@proton.me
   - For security issues
   - Private inquiries

## Common Error Messages

### `ModuleNotFoundError: No module named 'X'`
→ Install missing package: `pip install X`

### `OSError: [Errno 48] Address already in use`
→ Port in use, use `--port 8053`

### `PermissionError: [Errno 13] Permission denied`
→ Check file/folder permissions

### `ImportError: cannot import name 'X'`
→ Update Neural DSL: `pip install --upgrade neural-dsl`

### `SyntaxError: invalid syntax`
→ Check DSL syntax, use Parse button

### `TypeError: X() got unexpected keyword argument`
→ Check parameter names in DSL

### `ValueError: X is not supported`
→ Feature not available in selected backend

### `ResourceExhaustedError: OOM`
→ Reduce batch size or model size

## Prevention Tips

### Best practices to avoid issues

1. **Use virtual environments**
   - Isolates dependencies
   - Prevents conflicts

2. **Keep packages updated**
```bash
pip install --upgrade neural-dsl dash
```

3. **Parse before compiling**
   - Catches syntax errors early
   - Validates model structure

4. **Start simple**
   - Use examples as templates
   - Add complexity gradually

5. **Save frequently**
   - Export working models
   - Version your DSL files

6. **Test with small datasets**
   - Quick iterations
   - Catch errors faster

7. **Monitor resources**
   - Watch CPU/memory usage
   - Don't run too many processes

8. **Read error messages**
   - They usually tell you what's wrong
   - Google the exact error

## Still Having Issues?

If this guide didn't solve your problem:

1. **Check for updates**: `pip install --upgrade neural-dsl`
2. **Try example models**: Use "Load Example" button
3. **Simplify**: Remove complexity, isolate the issue
4. **Ask for help**: Use resources above

Remember: Most issues have simple solutions. Don't hesitate to ask for help!

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
