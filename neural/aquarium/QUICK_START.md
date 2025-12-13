# Quick Start Guide

## Installation

```bash
cd neural/aquarium
npm install
```

## Run Development Server

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

## Basic Usage

### 1. Add Your First Layer

**Option A: Drag and Drop**
- Find a layer in the left palette (e.g., "Dense" under "Core")
- Drag it onto the canvas
- Drop it where you want it

**Option B: Click to Add**
- Click a layer in the palette
- It appears on the canvas at a random position

### 2. Connect Layers

- Click and drag from the bottom handle of one layer
- Drop on the top handle of another layer
- Invalid connections will show an error message

### 3. Edit Layer Parameters

- Click a layer to select it
- The right panel shows its properties
- Edit parameters (units, activation, etc.)
- Changes apply immediately

### 4. View Generated Code

- Click "💻 Show Code" in the toolbar
- See the Neural DSL code
- Edit code directly
- Click "📊 Show Designer" to return

### 5. Save Your Work

- Click "💾 Export" to download a .neural file
- Click "📂 Import" to load a saved file

## Example Workflow: Build MNIST CNN

1. **Start with Input** (already on canvas)
   - Default: `(None, 28, 28, 1)`

2. **Add Conv2D**
   - Drag from "Convolutional" category
   - Connect to Input
   - Edit: `filters=32`, `kernel_size=[3,3]`, `activation="relu"`

3. **Add MaxPooling2D**
   - Drag from "Pooling" category
   - Connect to Conv2D
   - Default: `pool_size=[2,2]`

4. **Add another Conv2D**
   - `filters=64`, `kernel_size=[3,3]`, `activation="relu"`

5. **Add MaxPooling2D**
   - Connect to second Conv2D

6. **Add Flatten**
   - From "Core" category
   - Connects 2D output to 1D

7. **Add Dense**
   - `units=128`, `activation="relu"`

8. **Add Dropout**
   - `rate=0.5`

9. **Add Dense (Output)**
   - `units=10`, `activation="softmax"`

10. **View Code**
    - Click "💻 Show Code"
    - See your network in DSL format

11. **Export**
    - Click "💾 Export"
    - Save as `mnist_cnn.neural`

## Keyboard Shortcuts

- **Delete**: Remove selected node
- **Ctrl+C**: Copy (when node selected)
- **Ctrl+V**: Paste
- **Ctrl+Z**: Undo (planned)
- **Ctrl+S**: Export (planned)

## Tips & Tricks

### Organize Your Network
- Use "🔄 Auto Layout" to arrange layers vertically
- Drag layers to reposition manually
- Use the mini-map (bottom right) to navigate large networks

### Edit Efficiently
- Select a layer and edit parameters immediately
- Use Tab to move between parameter fields
- Arrays: `[3, 3]` or `(3, 3)` both work

### Connection Validation
- Red error = incompatible connection
- Hover over handles to see connection points
- Can't create cycles

### Code Editing
- Edit DSL directly in code view
- Changes sync to visual immediately
- Syntax errors won't crash (shows in console)

### Shape Propagation
- Output shapes calculated automatically
- Shown in each layer node
- Helps verify network structure

## Troubleshooting

### Layer won't connect
**Problem:** "Cannot connect X to Y"
**Solution:** Check layer compatibility (e.g., can't connect Flatten to Conv2D)

### Parameters not updating
**Problem:** Changes not visible
**Solution:** Click outside parameter field to apply

### Code view blank
**Problem:** No code shown
**Solution:** Add at least one layer after Input

### Import failed
**Problem:** File won't load
**Solution:** Ensure it's a valid .neural file with correct syntax

## Next Steps

- Explore all layer categories
- Try different network architectures
- Load examples from `examples/` folder
- Read DEVELOPER_GUIDE.md for advanced features
- Integrate with Neural DSL backend

## Example Models

Load these from the `examples/` folder:

- `mnist_cnn.neural` - Convolutional network for MNIST
- `lstm_text.neural` - Recurrent network for text classification

## Support

- Check README.md for feature documentation
- See DEVELOPER_GUIDE.md for technical details
- Report issues on GitHub (if applicable)
