# Aquarium IDE Quick Reference Card

<div align="center">

**Print this page for easy reference** 🖨️

</div>

---

## 🚀 Launch Commands

```bash
# Start Aquarium
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode
python -m neural.aquarium.aquarium --debug
```

**Access**: `http://localhost:8052`

---

## ⌨️ Essential Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Parse DSL | `Ctrl+P` | `⌘P` |
| Compile | `Ctrl+B` | `⌘B` |
| Run | `Ctrl+R` | `⌘R` |
| Stop | `Ctrl+C` | `⌘C` |
| Save | `Ctrl+S` | `⌘S` |
| Load Example | `Ctrl+E` | `⌘E` |

---

## 🎯 Workflow

```
1. Write DSL → 2. Parse → 3. Configure → 4. Compile → 5. Run
```

---

## 📝 DSL Template

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

---

## 🔧 Backend Selection

| Backend | Use Case |
|---------|----------|
| **TensorFlow** | Production, deployment |
| **PyTorch** | Research, flexibility |
| **ONNX** | Cross-platform |

---

## 📊 Dataset Options

| Dataset | Shape | Classes |
|---------|-------|---------|
| MNIST | (28,28,1) | 10 |
| CIFAR10 | (32,32,3) | 10 |
| CIFAR100 | (32,32,3) | 100 |
| Custom | User-defined | Any |

---

## ⚙️ Training Config

| Parameter | Range | Default |
|-----------|-------|---------|
| Epochs | 1-1000 | 10 |
| Batch Size | 1-2048 | 32 |
| Val Split | 0.0-1.0 | 0.2 |

---

## 🎨 Action Buttons

| Button | Function |
|--------|----------|
| **Parse DSL** | Validate model |
| **Compile** | Generate code |
| **Run** | Execute training |
| **Stop** | Terminate process |
| **Export** | Save script |
| **Clear** | Reset console |

---

## 🚦 Status Indicators

| Badge | Color | Meaning |
|-------|-------|---------|
| Idle | Gray | Ready |
| Compiled | Green | Ready to run |
| Running | Blue | Training |
| Error | Red | Failed |
| Stopped | Yellow | User stopped |

---

## 🐛 Common Errors

| Error | Solution |
|-------|----------|
| Port in use | Use `--port 8053` |
| Module not found | `pip install neural-dsl[dashboard]` |
| Parse error | Check DSL syntax |
| OOM | Reduce batch size |
| Dataset fails | Check internet, wait for download |

---

## 📦 Installation

```bash
# Quick install
pip install neural-dsl[dashboard]

# Full install
pip install neural-dsl[full]

# From source
git clone https://github.com/Lemniscate-world/Neural.git
pip install -e ".[dashboard]"
```

---

## 📚 Documentation Links

- **Full Manual**: [user-manual.md](user-manual.md)
- **Installation**: [installation.md](installation.md)
- **Shortcuts**: [keyboard-shortcuts.md](keyboard-shortcuts.md)
- **Troubleshooting**: [troubleshooting.md](troubleshooting.md)

---

## 💡 Tips

✅ **DO**:
- Parse before compiling
- Start with small epochs
- Use examples as templates
- Export successful models

❌ **DON'T**:
- Skip parsing step
- Use huge batch sizes on small datasets
- Forget to select correct dataset
- Run multiple models simultaneously

---

## 🆘 Get Help

- **Docs**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Chat**: [Discord](https://discord.gg/KFku4KvS)
- **Email**: Lemniscate_zero@proton.me

---

<div align="center">

**Aquarium IDE v0.1.0**

[🌟 Star on GitHub](https://github.com/Lemniscate-world/Neural) • 
[📚 Full Docs](README.md) • 
[💬 Community](https://discord.gg/KFku4KvS)

</div>

---

**Print-Friendly Version**: Press Ctrl+P / Cmd+P to print this reference card

---

**Last Updated**: December 2024
