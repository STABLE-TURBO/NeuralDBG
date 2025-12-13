# Neural Aquarium IDE 🐠

*A web-based IDE for Neural DSL that I built because I was tired of switching between text editors and terminals. Now I can write models, train them, and watch them learn—all in one place.*

## What This Actually Does

You know that feeling when you're working on a neural network and you're constantly jumping between your editor, terminal, and browser to check results? Yeah, I got tired of that. So Aquarium brings everything together: write your DSL code, click a button, and watch your model train with real-time metrics. It's basically "what if VS Code and TensorBoard had a baby that spoke Neural DSL?"

**TL;DR**: Point your browser at `localhost:8052` and you've got a complete DSL development environment. No more terminal juggling.

## Starting Up

```bash
python -m neural.aquarium.aquarium
```

That's it. Open `http://localhost:8052` and you're in. Want a different port? Tack on `--port 8053`. Need to debug the IDE itself? Use `--debug` (meta, I know).

## The First Time I Used This

Here's what actually happens when you fire up Aquarium:

**Left side**: A code editor where you write Neural DSL. It's got syntax highlighting (took way too long to get that working with Dash, but worth it). Below that, example models you can load with one click because nobody wants to start from a blank page.

**Right side**: The "mission control" panel. Backend selection (TensorFlow/PyTorch/ONNX), dataset picker, and all your training knobs (epochs, batch size, etc.). Below that, a console that shows you what's happening in real-time, and metrics charts that update as your model trains.

The workflow feels like this:
1. Write or load a model
2. Click "Parse DSL" to make sure you didn't typo anything
3. Pick your backend and dataset
4. Hit "Compile" to generate Python code
5. Hit "Run" and watch it go

**Pro tip**: Keep the console visible. It's oddly satisfying watching the training progress bars scroll by.

## The Cool Parts (That I'm Proud Of)

### Live Compilation & Execution

This was the hardest part to get right. When you hit "Run", Aquarium spawns a separate Python process (because blocking the main thread would freeze the UI), captures stdout/stderr in real-time, and streams it to the console. It parses the output for metrics (loss, accuracy) and updates the charts live.

The technical bit: I'm using `subprocess.Popen` with `stdout=PIPE` and a separate thread that reads output line-by-line. Had to be careful with buffering—unbuffered mode (`python -u`) is your friend here.

### Backend Switching Without Rewriting Code

You write DSL once, and Aquarium can generate TensorFlow, PyTorch, or ONNX code from the same source. Just change the dropdown. This is possible because the code generator is abstracted—it parses your DSL into an AST, then different backends consume that AST.

I added this because I kept forgetting PyTorch syntax and would waste time translating TensorFlow code. Now I just click a button.

### Dataset Handling (The Pragmatic Way)

Aquarium knows about common datasets (MNIST, CIFAR10, CIFAR100, ImageNet) and generates the loading code for you. But here's the thing: I didn't want to limit you to built-ins, so there's a "Custom" option where you can point to your own data.

**Work-in-progress note**: The custom dataset path assumes a specific directory structure (train/val folders with class subdirectories). I want to make this more flexible—maybe auto-detect common formats or let you provide a loading script. On the TODO list.

### Script Export (Because the IDE Won't Always Be Open)

After you compile, you can export the generated Python script. Click "Export Script", give it a name, pick a folder, done. The exported script is completely standalone—it has all the imports, dataset loading, model definition, training loop, and saving logic. You can run it directly with `python my_model.py`.

This is intentional. I wanted Aquarium to be a development tool, not a prison. You should be able to take your code and run with it.

## Real Example: Training an MNIST Classifier

Let me walk you through what I do when I'm iterating on a new architecture:

**Step 1**: Load the MNIST example from the sidebar. It's a simple ConvNet:

```neural
network MNISTClassifier {
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

**Step 2**: Click "Parse DSL". If you see green text in the console saying "DSL parsed successfully", you're good. If not, fix your typos (I constantly forget commas).

**Step 3**: Configure the run:
- Backend: TensorFlow (it's what I know best)
- Dataset: MNIST (auto-loads from Keras)
- Epochs: 5 (for quick iteration; bump to 10-15 for real training)
- Batch size: 32 (sweet spot for my laptop's RAM)
- Validation split: 0.2 (saves 20% of training data for validation)

**Step 4**: Hit "Compile". Watch the console—it should show "Compilation successful!" and display the generated Python code path.

**Step 5**: Hit "Run". Now the fun part. You'll see:
```
Epoch 1/5
1500/1500 [==============================] - 12s 8ms/step - loss: 0.1234 - accuracy: 0.9621 - val_loss: 0.0523 - val_accuracy: 0.9823
```

And the metrics charts start updating. Loss goes down, accuracy goes up. Dopamine hit every time.

**Step 6** (optional): If you want to tweak and re-run, just edit the DSL and hit "Compile" → "Run" again. If you like what you see, hit "Export Script" to save it.

## Architecture Decisions (And Why)

### Why Dash Instead of Flask + React?

I initially prototyped with Flask and vanilla JavaScript. It worked, but every UI change required writing HTML, CSS, and JS by hand. Dash lets me build the UI in Python with a component model, which is way faster for prototyping. Yes, it's less flexible than React, but for an internal tool, the productivity win is worth it.

**Trade-off**: Dash's callback system can feel weird if you're used to traditional request/response. You define `@callback` decorators that specify inputs/outputs, and Dash handles the wiring. Took me a day to stop fighting it and start thinking in callbacks.

### Why Separate Processes for Execution?

Security and stability. If I ran training code in the same process as the web server, a crashed model would kill the whole IDE. Also, model training can block—you want that in a separate process so the UI stays responsive.

I'm using Python's `subprocess` module with `Popen` for fine-grained control (can capture output, kill processes, set environment variables). It's lower-level than `subprocess.run()`, but you need that control for a good UX.

### Why Not Use a Terminal Emulator Component?

I tried! There are web-based terminal emulators (xterm.js, etc.), but they're heavy and overkill for showing read-only output. The current console is just a `<pre>` tag with auto-scroll. Simple, fast, works.

**Limitation**: It can't handle interactive input. If your training script asks for user input, it'll hang. I catch common cases (dataset downloads that prompt for confirmation) and auto-answer, but it's not foolproof.

## The Components (Under the Hood)

If you're poking around the code, here's what does what:

**`aquarium.py`**: Entry point. Sets up the Dash app, registers callbacks, starts the dev server. It's about 200 lines—the actual UI logic is split into components.

**`runner_panel.py`**: The entire right-side panel. All the dropdowns, buttons, console, and charts. Uses Dash Bootstrap for layout (because I can't design to save my life). Returns the UI tree and handles none of the business logic—that's in ExecutionManager.

**`execution_manager.py`**: The brains. Methods like `compile_model()`, `run_script()`, `stop_execution()`. This is where subprocess magic happens. Also parses metrics from console output using regex (because TensorFlow's progress bars are *just* structured enough to parse, but not structured enough to make it easy).

**`script_generator.py`**: Takes compiled DSL code and wraps it in a complete training script. Generates imports, dataset loading, model building, training loop, evaluation, and saving. Different backends need different wrapping code (e.g., PyTorch uses `DataLoader`, TensorFlow uses `model.fit()`), so there's some conditional logic here.

## What I'm Still Working On

**Syntax highlighting** (✅ Done): Used Dash's code editor component with a custom Neural DSL syntax definition. Was a pain, but it works now.

**Auto-complete** (❌ Not done): I want IntelliSense-style suggestions. The parser knows what layers/activations are valid, so this is doable. Just need to wire up the completion API. Problem: Dash's editor component doesn't expose completion hooks easily. Might need to switch to CodeMirror 6 or Monaco.

**Hyperparameter optimization** (⚠️ Partially done): There's an HPO checkbox that passes `--hpo` to the CLI, which triggers Optuna-based tuning. It works, but the UI doesn't show Optuna's study dashboard—you just see logs. I want to embed Optuna's visualization or at least show trials in a table.

**Experiment tracking** (❌ Not done): Right now, every run is isolated. If you train 10 models, you have no easy way to compare them. I want to add a history panel that tracks runs (hyperparams, metrics, timestamps) and lets you diff models. Thinking about using SQLite for storage.

**Collaborative editing** (❌ Not done, probably won't happen): Would be cool if multiple people could edit the same DSL file in real-time. But this is a *lot* of work (need WebSockets, OT or CRDT for conflict resolution, user management). Probably overkill for a DSL IDE. If you need this, just use Git.

**Cloud execution** (❌ Not done): Local execution is fine for small models, but if you're training ResNet-50, you want a GPU cluster. I'm thinking about adding a "Run on Cloud" button that submits to SageMaker/Vertex AI/etc. The `neural/integrations/` folder already has connectors, so it's mostly UI work.

## Troubleshooting (Real Issues I've Hit)

### "Address already in use" when starting Aquarium

Someone's using port 8052 (maybe an old Aquarium process you forgot to kill?). Try:
```bash
python -m neural.aquarium.aquarium --port 8053
```

Or find and kill the process hogging 8052:
```bash
# Windows
netstat -ano | findstr :8052
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8052 | xargs kill -9
```

### Console shows "ModuleNotFoundError: No module named 'tensorflow'"

You're trying to use TensorFlow but didn't install it. Either switch to a backend you have installed, or:
```bash
pip install tensorflow
# Or install everything:
pip install -e ".[full]"
```

### Training starts but console shows nothing

Buffering issue. The script is running but output isn't being flushed. This shouldn't happen (I use `-u` flag for unbuffered output), but if it does, check the exported script and add `sys.stdout.flush()` calls after print statements.

### Charts aren't updating

Metrics parsing might be failing. Open the browser console (F12) and look for JavaScript errors. Also check the Aquarium terminal—I log parsing failures there. If your training output format is non-standard, the regex might not match.

### "Run" button does nothing

Check the Aquarium terminal for Python exceptions. Common causes:
- Compilation failed (but UI didn't show it clearly)
- Generated script has a syntax error (bug in script generator)
- Python interpreter not found (rare, but possible in virtual envs)

## Contributing (If You're Into That)

I built Aquarium for myself, but if you want to add features:

1. **Code style**: Match what's there (100-char lines, type hints, PEP 8). Run `ruff check .` before committing.
2. **Test stuff**: I don't have great test coverage for the UI (mocking Dash callbacks is annoying), but at least manually test your changes. Don't break existing examples.
3. **Documentation**: Update this README if you add a major feature. I like the personal/story-driven style, so keep that vibe.
4. **UI changes**: Keep it simple. Aquarium is a developer tool, not a design showcase. Function over form.

If you add a feature I like, I'll merge it. If it's half-baked or breaks existing stuff, I'll ask you to fix it. Standard open-source vibes.

## The Bigger Picture

Aquarium is part of the Neural DSL project, which is trying to make neural network development less painful. The DSL lets you define models at a high level, and Aquarium gives you a place to iterate on those models quickly.

Other tools in the ecosystem:
- **Neural CLI** (`neural compile`, `neural run`, etc.): Command-line interface for DSL operations
- **NeuralDbg** (`neural/dashboard/`): Real-time debugger with layer visualization (port 8050)
- **No-code GUI** (`neural/no_code/`): Drag-and-drop model builder (port 8051)

Aquarium sits between the CLI (too manual) and the no-code GUI (too restrictive). It's for people who want to write code but don't want the ceremony of managing scripts.

## License

Same as the Neural DSL package—check the repo root for details.

---

*Built with Dash, caffeine, and the stubborn belief that development tools should feel good to use.*
