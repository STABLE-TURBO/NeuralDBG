# Troubleshooting Guide

Look, we've all been there—staring at an error message at 2 AM, wondering why the DSL that worked yesterday suddenly doesn't. This guide is here to help you debug those moments with real stories from the community and practical fixes that actually work.

## Table of Contents

- [Installation Headaches](#installation-headaches)
- [Parser Errors (The "Why Won't This Parse?" Edition)](#parser-errors-the-why-wont-this-parse-edition)
- [Shape Propagation Nightmares](#shape-propagation-nightmares)
- [Code Generation Weirdness](#code-generation-weirdness)
- [Runtime Surprises](#runtime-surprises)
- [Dashboard Not Cooperating](#dashboard-not-cooperating)
- [HPO Taking Forever](#hpo-taking-forever)
- [Cloud Integration Blues](#cloud-integration-blues)
- [Performance Slowdowns](#performance-slowdowns)
- [Getting Help (We're Here For You)](#getting-help-were-here-for-you)

---

## Installation Headaches

### "pip install just won't work and I'm losing my mind"

Yeah, dependency conflicts are the worst. You just want to get started, and instead you're debugging pip.

**What I tried first:** Just running `pip install neural-dsl` in my existing environment.

**What actually worked:**

Start fresh with a clean virtual environment. I know it feels like overkill, but it saves so much time:

```bash
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install neural-dsl
```

**Real story from @sarah_ml:** "I wasted 3 hours trying to fix conflicts with my existing TensorFlow install. Clean venv fixed it in 2 minutes. Just do it."

If you're still stuck, try installing without optional dependencies first, then add what you need:

```bash
pip install neural-dsl --no-deps
pip install lark click  # Just the essentials
```

Or go straight for the backend you know you'll use:

```bash
# For TensorFlow folks
pip install neural-dsl tensorflow

# For PyTorch fans
pip install neural-dsl torch
```

### "It installed but Python can't find it??"

This one's frustrating because pip says success but Python says "never heard of it."

**What I tried first:** Restarting my terminal (classic move).

**The actual problem:** Usually wrong Python environment or path issues.

Quick sanity check:

```bash
pip show neural-dsl
python -c "import neural; print(neural.__version__)"
```

If that fails, check which Python you're actually using:

```bash
which python  # Linux/macOS
where python  # Windows
```

**From @devops_dan:** "Had this happen because I was using system Python while pip installed to my venv. Use `python -m pip` instead of just `pip` to be sure."

For development work, editable mode is your friend:

```bash
pip install -e .
```

### "Version mismatch warnings everywhere"

Not the end of the world, but annoying enough that you want to fix it.

**What I tried first:** Ignoring it (worked great until it didn't).

**Better solution:**

```bash
pip install --upgrade neural-dsl
```

If that doesn't help, nuke it from orbit:

```bash
pip cache purge
pip install --force-reinstall neural-dsl
```

---

## Parser Errors (The "Why Won't This Parse?" Edition)

### "Unexpected token errors that make no sense"

Parser errors are notoriously cryptic. You know your DSL looks right, but the parser disagrees.

**What I tried first:** Staring at the line it complained about for 10 minutes.

**What actually helps:** Look at the line *before* the error. Seriously.

Missing commas or colons are the usual suspects:

```yaml
# This will error on line 2 (not line 1!)
network MyModel {
  input (28, 28, 1)  # Missing the colon
  
# Fixed version
network MyModel {
  input: (28, 28, 1)
```

**Real debugging story from @ml_newbie:** "Got 'Unexpected token Dense' on line 5. Spent an hour looking at Dense. The problem? Missing comma in the Conv2D on line 4. Parser error line numbers are... suggestions."

Indentation is also surprisingly picky:

```yaml
# This drives the parser crazy
network MyModel {
  input: (28, 28, 1)
  layers:
     Dense(64)  # 4 spaces
   Dense(32)    # 2 spaces - parser hates you now

# Pick a spacing and stick with it
network MyModel {
  input: (28, 28, 1)
  layers:
    Dense(64)
    Dense(32)
```

### "Unknown layer type" (but I KNOW it exists)

This error is annoying because you're 100% sure Conv2D is a thing.

**What I tried first:** Checking the docs to confirm I'm not going insane.

**The gotcha:** It's almost always a capitalization issue.

```yaml
# ❌ Nope
Conv2d(32, (3,3))
conv2d(32, (3,3))

# ✅ Yes
Conv2D(32, (3,3))
```

**From @pytorch_paul:** "Came from PyTorch where it's Conv2d. Took me embarrassingly long to realize Neural DSL wants Conv2D with capital D."

Not sure what layers are available? Check the list:

```bash
neural layers --list
```

For custom layers, make sure you define them before using them:

```yaml
# Define it
define CustomBlock {
  Conv2D(32, (3,3), "relu")
  BatchNormalization()
}

# Then use it
network MyModel {
  layers:
    CustomBlock()
}
```

### "Parameter validation failed"

The parser is telling you your numbers don't make sense, and it's actually trying to help.

**What I tried first:** Double-checking my math.

**Common mistakes:**

```yaml
# This is just wrong
Dense(units=-5)  # Negative neurons? Really?
Dropout(rate=1.5)  # That's 150% dropout my friend

# Parameter rules to remember:
# - Dense.units > 0
# - Dropout.rate between 0 and 1
# - Conv2D.filters > 0
# - kernel_size > 0
```

Also watch your data types:

```yaml
# Parser wants integers for units
Dense(units=64.5)  # ❌
Dense(units=64)    # ✅
```

**From @hyperparameter_harry:** "Got validation errors on my HPO ranges. Had min and max backwards. `HPO(range(100, 10))` makes no sense—start small, go big."

---

## Shape Propagation Nightmares

### "Shape mismatch and I have no idea why"

This is probably the most common error you'll hit, and yeah, it's frustrating.

**What I tried first:** Checking the shapes manually with a calculator.

**What actually helps:** Understanding what each layer expects.

The classic mistake—going from Conv2D straight to Dense:

```yaml
# This will explode
Conv2D(64, (3,3))
Dense(128)  # Dense wants 2D, Conv2D outputs 4D

# The fix everyone learns eventually
Conv2D(64, (3,3))
Flatten()  # Converts 4D -> 2D
Dense(128)
```

**Real story from @cnn_christine:** "Hit this on my first model. Error said 'Expected (None, 128) got (None, 32, 32, 64)'. Took me way too long to realize I needed Flatten. Now it's muscle memory."

Alternative approach using GlobalAveragePooling:

```yaml
Conv2D(64, (3,3))
GlobalAveragePooling2D()  # Another way to flatten
Dense(128)
```

### "Input shape format is apparently wrong?"

The error message here is not super helpful, but the fix is straightforward.

**What I tried first:** Guessing different formats until something worked.

**What you need to know:**

```yaml
# ❌ Not a tuple
input: 784

# ✅ Tuple for 1D
input: (784,)

# ✅ Tuple for images
input: (28, 28, 1)

# ✅ Explicit batch dimension (optional)
input: (None, 28, 28, 1)
```

**From @shape_shifter:** "Remember TensorFlow uses channels-last (28, 28, 1) while PyTorch uses channels-first (1, 28, 28). Neural DSL handles the conversion, just pick one and go."

### "Can't generate shape flow diagram"

Usually this means your model structure has issues that need fixing first.

**What I tried first:** Trying to visualize it anyway (narrator: it didn't work).

**Better approach:** Validate before visualizing.

```bash
# Check your model is valid first
neural compile model.neural --dry-run

# Then visualize
neural visualize model.neural
```

Watch out for circular dependencies in macros:

```yaml
# This is a problem
define A {
  B()
}
define B {
  A()  # A calls B, B calls A... infinite loop
}
```

**From @viz_victor:** "Got weird visualization errors. Turned out I had typos in my macro names that created accidental recursion. Dry-run validation caught it."

---

## Code Generation Weirdness

### "Backend doesn't support my layer"

Not all layers work with all backends, which is annoying when you find out the hard way.

**What I tried first:** Switching to a different layer name (didn't help).

**What works:** Check support before committing to a layer.

```bash
neural layers --backend pytorch
neural layers --backend tensorflow
```

If your layer isn't supported, you have options:

```yaml
# If TransformerEncoder doesn't work in PyTorch, build it yourself:
layers:
  MultiHeadAttention(num_heads=8)
  Dense(2048, activation="relu")
  Dense(512)
```

Or just switch backends:

```bash
neural compile model.neural --backend tensorflow
```

**From @backend_betty:** "Needed LSTM for PyTorch. Not supported. Switched to TensorFlow backend, worked instantly. Sometimes it's easier to change backends than rewrite your model."

### "Generated code has syntax errors"

This one's usually a bug on our end, and we want to know about it.

**What I tried first:** Manually fixing the generated code (tedious).

**Better approach:**

Update first:

```bash
pip install --upgrade neural-dsl
```

If it's still broken, please report it:

```bash
# Create a minimal DSL that reproduces the issue
# Then hit us up: https://github.com/Lemniscate-world/Neural/issues
```

**From @code_gen_charlie:** "Found a bug where optimizer parameters weren't generating. Updated to latest version, fixed. Always update first before assuming it's your fault."

---

## Runtime Surprises

### "CUDA out of memory (the classic)"

This one never stops being annoying, even when you know how to fix it.

**What I tried first:** Closing everything else on my computer (didn't help).

**What actually helps:**

Lower your batch size—it's almost always the fix:

```yaml
train {
  batch_size: 16  # Was 32? Try 16. Still OOM? Try 8.
  epochs: 15
}
```

**Real story from @gpu_poor_sam:** "Tried training ResNet with batch_size=64 on my 1080Ti. LOL no. Dropped to 8, added gradient accumulation to compensate. Works great now."

Gradient accumulation lets you keep effective batch size large:

```python
# In generated code, effective batch_size = actual_batch_size * accumulation_steps
for i, (x, y) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Or enable mixed precision (basically free performance):

```bash
neural compile model.neural --mixed-precision
```

### "NaN loss (the silent killer)"

Everything looks fine, then suddenly: nan. Training ruined.

**What I tried first:** Panicking and restarting training (waste of time).

**Debugging approach:**

Lower your learning rate:

```yaml
optimizer: Adam(learning_rate=0.0001)  # Was 0.001? Go smaller
```

Add gradient clipping (saves lives):

```yaml
train {
  gradient_clip: 1.0  # Prevents explosion
}
```

**From @nan_nightmare_nina:** "Hit NaN on epoch 3. Added gradient clipping and BatchNorm. Never saw NaN again. BatchNorm is like insurance."

BatchNormalization helps stability:

```yaml
layers:
  Conv2D(32, (3,3))
  BatchNormalization()  # Add this after Conv2D
  Activation("relu")
```

NeuralDbg can catch this early:

```bash
neural debug model.neural --anomalies
```

### "Model not learning anything"

Loss is high, stays high, won't budge. Feels bad.

**What I tried first:** Training longer (didn't help, just wasted time).

**Actual solutions:**

Try a learning rate schedule:

```yaml
optimizer: Adam(learning_rate=0.001)
lr_schedule: ExponentialDecay(
  initial_lr=0.001,
  decay_steps=1000,
  decay_rate=0.96
)
```

Check your data preprocessing (super common issue):

```python
# Images need normalization
X = X / 255.0

# Other data too
X = (X - mean) / std
```

**Real story from @stuck_steve:** "Trained for 50 epochs, nothing. Realized I forgot to normalize my input images. Normalized them, model converged in 5 epochs. Check your data first."

Maybe your model is too small:

```yaml
Dense(256)  # Was 128? Go bigger
```

Or try different activations:

```yaml
Dense(128, activation="elu")  # relu not working? Try elu or selu
```

---

## Dashboard Not Cooperating

### "Dashboard won't start"

You're excited to debug, but the dashboard refuses to launch.

**What I tried first:** Running the command again (didn't help).

**The usual culprit:** Port 8050 is taken.

```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8050

# Linux/macOS:
lsof -i :8050
```

Just use a different port:

```bash
neural debug model.neural --port 8051
```

**From @port_conflict_pete:** "Had Jupyter on 8050. Dashboard tried to use 8050. Chaos. Now I always specify --port."

Also check your firewall—it might be blocking Python.

### "Dashboard opens but shows nothing"

The UI loads, but it's just... empty. Sad times.

**What I tried first:** Refreshing the page repeatedly (optimistic).

**What actually works:**

Make sure your model is actually running:

```bash
neural debug model.neural --execute
```

Check browser console (F12) for WebSocket errors. If the connection drops, reload.

Verify tracing is enabled in your generated code:

```python
from neural.dashboard import trace_execution
trace_execution(enabled=True)
```

### "Real-time updates stopped"

It was working, now it's frozen. Classic.

**What I tried first:** Waiting patiently (narrator: patience didn't help).

**Quick fixes:**

Hard reload your browser:

```
Ctrl+F5  # Windows/Linux
Cmd+Shift+R  # macOS
```

Or clear cache:

```
Ctrl+Shift+Delete → Clear cache
```

**From @realtime_rachel:** "Updates stopped after I switched tabs. Turned out browser was throttling background tabs. Kept dashboard tab active, problem solved."

Update Neural DSL if you're on an old version:

```bash
pip install --upgrade neural-dsl
```

---

## HPO Taking Forever

### "Search space is massive and I don't have infinite time"

HPO can take forever if you're not careful about your search space.

**What I tried first:** Just letting it run overnight (it was still running in the morning).

**Smarter approach:**

Reduce your search space:

```yaml
# This is way too many options
Dense(units=HPO(range(10, 1000, step=10)))  # 100 trials!

# This is reasonable
Dense(units=HPO(choice(32, 64, 128, 256)))  # 4 trials
```

Add early stopping:

```yaml
hpo_config {
  trials: 50
  early_stopping: 5  # Give up if no improvement for 5 trials
}
```

**From @hpo_henry:** "First HPO run took 2 days. Learned to use choice() instead of range(), added early stopping. Now it takes 2 hours."

Run trials in parallel if you can:

```bash
neural hpo model.neural --parallel 4
```

### "HPO fails with parameter errors"

HPO syntax is a bit particular, and the errors aren't always clear.

**What I tried first:** Copy-pasting from examples (worked better than guessing).

**Correct syntax:**

```yaml
# These work
learning_rate=HPO(log_range(1e-4, 1e-2))
units=HPO(choice(32, 64, 128))
rate=HPO(range(0.1, 0.9, step=0.1))
```

Not sure what supports HPO?

```bash
neural hpo --show-params
```

### "Best parameters found but not applied"

HPO finished, found great params, but your model is still using the old ones?

**What I tried first:** Manually copying params from the output (error-prone).

**Automatic way:**

```bash
neural hpo model.neural --output optimized.neural
```

Or check what it found:

```bash
neural hpo model.neural --show-best
```

---

## Cloud Integration Blues

### "Can't connect to cloud platform"

Cloud stuff should just work, but sometimes it doesn't.

**What I tried first:** Checking my internet connection (it was fine).

**Actual problem:** Usually credentials.

For Kaggle:

```bash
cat ~/.kaggle/kaggle.json  # Does this exist and look right?
```

For AWS:

```bash
aws configure list
```

Make sure you installed cloud dependencies:

```bash
pip install neural-dsl[cloud]
```

**From @cloud_carl:** "Spent an hour debugging Kaggle connection. Kaggle credentials were in wrong location. Move them to ~/.kaggle/ and chmod 600 them."

### "ngrok tunnel won't start"

Tunneling is magic when it works, frustrating when it doesn't.

**What I tried first:** Running the command multiple times (didn't help).

**Solution:** Install ngrok properly:

```bash
# Windows
choco install ngrok

# Linux/macOS
brew install ngrok
```

Set your auth token:

```bash
ngrok authtoken YOUR_TOKEN
```

Or try an alternative:

```bash
neural cloud run --tunnel localtunnel
```

**From @tunnel_tom:** "ngrok kept timing out. Switched to localtunnel, worked first try. Sometimes the alternative is just better."

---

## Performance Slowdowns

### "Compilation takes forever"

You just want to compile your DSL, not wait 5 minutes.

**What I tried first:** Making coffee while it compiled (at least I got coffee).

**Speed it up:**

```bash
# Cache compilation results
neural compile model.neural --cache

# Skip validation if you're confident
neural compile model.neural --no-validate

# See where time is spent
neural compile model.neural --profile
```

### "Training is painfully slow"

Your training loop is crawling along and you're losing your mind.

**What I tried first:** Blaming my hardware (sometimes fair).

**Actual optimizations:**

Enable GPU if you have one:

```yaml
device: "cuda"  # Or "auto" to auto-detect
```

Optimize data loading:

```python
DataLoader(..., num_workers=4, pin_memory=True)
```

**Real story from @slow_no_more_sarah:** "Training was super slow. Realized I was using CPU when I had a GPU. Set device='cuda', 10x speedup instantly. Always check your device."

Profile to find bottlenecks:

```bash
neural profile model.neural
```

---

## Getting Help (We're Here For You)

### Before you ask (saves everyone time)

Check your version:

```bash
neural --version
pip show neural-dsl
```

Create a minimal example—strip your DSL down to the smallest thing that reproduces the issue. Makes it way easier for us to help.

Collect error logs:

```bash
neural compile model.neural --verbose > debug.log 2>&1
```

### Where to find us

**Documentation:**
- [DSL Reference](dsl.md) - syntax and layer details
- [CLI Reference](cli.md) - all the commands
- [Examples](../examples/README.md) - working code to learn from

**Community:**
- [Discord Server](https://discord.gg/KFku4KvS) - fastest for real-time help
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions) - for longer conversations
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues) - for bugs and feature requests

### How to write a good bug report

This format helps us help you faster:

```markdown
**Environment:**
- Neural DSL: 0.2.9
- Python: 3.9.7
- OS: Windows 10
- Backend: TensorFlow 2.12.0

**What I'm trying to do:**
Train a simple CNN on MNIST

**What's happening:**
Getting shape mismatch error when using Flatten after Conv2D

**DSL Code:**
[Paste your minimal DSL here]

**Error Message:**
[Full traceback here]

**What I tried:**
- Added Flatten layer (still failed)
- Checked input shapes (seem correct)
- Googled the error (found nothing)

**Expected vs Actual:**
Expected: Should compile successfully
Actual: Shape mismatch error on line 8
```

**From @helpful_helen:** "Started writing detailed bug reports with what I tried. Got solutions 10x faster. People can't help if they don't understand the problem."

---

## Quick Reference: Common Errors

| Error | What it usually means | Try this first |
|-------|----------------------|----------------|
| `Unexpected token` | Syntax issue (probably previous line) | Check for missing commas/colons |
| `Unknown layer type` | Typo in layer name | Check capitalization (Conv2D not Conv2d) |
| `Shape mismatch` | Layers don't connect | Add Flatten between Conv and Dense |
| `Parameter validation failed` | Number out of bounds | Check constraints (dropout 0-1, units > 0) |
| `CUDA out of memory` | GPU is full | Lower batch size |
| `ModuleNotFoundError` | Missing dependency | Install backend: pip install tensorflow/torch |
| `HPO parameter error` | Wrong HPO syntax | Use choice() or range() correctly |
| `Backend not supported` | Feature not available | Try different backend or alternative layer |

---

## Quick Diagnostics (Run Through This)

- [ ] Am I on the latest version? (`pip install --upgrade neural-dsl`)
- [ ] Am I using Python 3.8 or newer?
- [ ] Is my virtual environment activated?
- [ ] Do I have TensorFlow or PyTorch installed?
- [ ] Does my DSL syntax look right? (commas, colons, indentation)
- [ ] Are my input shapes tuples? (e.g., `(28, 28, 1)` not `28, 28, 1`)
- [ ] Do my layer parameters make sense? (positive numbers, rates 0-1)
- [ ] Did I add Flatten before Dense after Conv layers?
- [ ] Is port 8050 available? (for dashboard)
- [ ] Do I have enough GPU/RAM?
- [ ] Can I reach the internet? (for cloud features)

---

**Still stuck?** Jump in our [Discord](https://discord.gg/KFku4KvS)—we've got a great community that loves solving these puzzles. No question is too basic. We've all been there.
