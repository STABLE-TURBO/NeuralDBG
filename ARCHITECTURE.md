# Neural DSL Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Neural DSL CLI                          │
│  neural compile │ neural run │ neural server start │ ...        │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌───────────┐   ┌──────────┐   ┌──────────┐
        │   Core    │   │ Extended │   │Experiment│
        │ Features  │   │ Features │   │  Features│
        └───────────┘   └──────────┘   └──────────┘
```

## Core Architecture (Always Enabled)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Core Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  .neural file                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  Parser  │───▶│    Shape     │───▶│     Code     │        │
│  │  (Lark)  │    │ Propagation  │    │  Generation  │        │
│  └──────────┘    └──────────────┘    └──────────────┘        │
│                                              │                  │
│                        ┌─────────────────────┼─────────┐       │
│                        ▼                     ▼         ▼       │
│                   tensorflow.py         pytorch.py   onnx.py   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Unified Server Architecture (v0.3.0+)

```
┌─────────────────────────────────────────────────────────────────┐
│              Unified Web Interface (Port 8050)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Navigation Bar                         │  │
│  │    Debug  │  Build  │  Monitor  │  Settings             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Tab Content                           │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │ Debug Tab    │  │ Build Tab    │  │ Monitor Tab  │ │  │
│  │  │ (NeuralDbg)  │  │ (No-Code)    │  │ (Monitoring) │ │  │
│  │  │              │  │              │  │              │ │  │
│  │  │ • Execution  │  │ • Layer      │  │ • Metrics    │ │  │
│  │  │   Trace      │  │   Palette    │  │ • Drift      │ │  │
│  │  │ • Resource   │  │ • Canvas     │  │ • Quality    │ │  │
│  │  │   Monitor    │  │ • Properties │  │ • Alerts     │ │  │
│  │  │ • Profiling  │  │ • Code Gen   │  │ • SLOs       │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend Services                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Parser    │  │   Shape     │  │    Code     │           │
│  │   Service   │  │   Service   │  │  Generator  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │     HPO     │  │   AutoML    │  │  Profiler   │           │
│  │  (Optuna)   │  │    (NAS)    │  │             │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Legacy Architecture (Deprecated, v0.2.x)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Separate Services (Deprecated)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Port 8050:  neural/dashboard/dashboard.py                     │
│              ├─ NeuralDbg Dashboard                            │
│              └─ Real-time monitoring                           │
│                                                                 │
│  Port 8051:  neural/no_code/no_code.py                        │
│              ├─ No-Code Builder                               │
│              └─ Visual editor                                  │
│                                                                 │
│  Port 8052:  neural/monitoring/dashboard.py                   │
│              ├─ Production Monitoring                         │
│              └─ Drift detection                               │
│                                                                 │
│  Port 8000:  neural/aquarium/backend/server.py                │
│              ├─ Aquarium API                                  │
│              └─ IDE backend                                   │
│                                                                 │
│  Port 5001:  Internal WebSocket (dashboard)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

      ⚠️  This architecture is deprecated and will be
          removed in v0.4.0. Use unified server instead.
```

## Feature Organization

```
┌─────────────────────────────────────────────────────────────────┐
│                        Feature Layers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Core Layer                            │  │
│  │  • Parser (Lark)                                        │  │
│  │  • Code Generation (TF, PyTorch, ONNX)                 │  │
│  │  • Shape Propagation                                    │  │
│  │  • CLI Interface                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┼──────────────────────────────┐  │
│  │                    Extended Layer                        │  │
│  │  • Debug Dashboard (NeuralDbg)                          │  │
│  │  • HPO (Hyperparameter Optimization)                    │  │
│  │  • AutoML (Neural Architecture Search)                  │  │
│  │  • Integrations (Cloud platforms)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┼──────────────────────────────┐  │
│  │                 Experimental Layer                       │  │
│  │  • No-Code Builder (Visual editor)                      │  │
│  │  • Monitoring (Production monitoring)                   │  │
│  │  • Marketplace (Model sharing)                          │  │
│  │  • Aquarium (Full IDE) [To be extracted]               │  │
│  │  • Federated (Distributed learning)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Compilation Flow
```
.neural file
    │
    ▼
┌──────────┐
│  Parser  │  Lark-based DSL parser
└────┬─────┘
     │ AST (Abstract Syntax Tree)
     ▼
┌──────────────┐
│  Transform   │  ModelTransformer converts AST to model_data
└────┬─────────┘
     │ model_data (dict)
     ▼
┌──────────────┐
│    Shape     │  Validates shapes, detects issues
│ Propagation  │  Suggests optimizations
└────┬─────────┘
     │ shape_history, validation results
     ▼
┌──────────────┐
│     Code     │  Generates backend-specific code
│  Generation  │  Applies optimizations
└────┬─────────┘
     │
     ├─▶ TensorFlow code (model_tensorflow.py)
     ├─▶ PyTorch code (model_pytorch.py)
     └─▶ ONNX code (model_onnx.py)
```

### Debug Flow
```
.neural file
    │
    ▼
Compilation → Generated code
    │
    ▼
Execution with instrumentation
    │
    ├─▶ Layer execution times
    ├─▶ Memory usage
    ├─▶ Gradient flow
    ├─▶ Dead neurons
    └─▶ Anomalies
    │
    ▼
WebSocket stream
    │
    ▼
Debug Dashboard (Port 8050/debug)
    │
    ├─▶ Real-time graphs
    ├─▶ Resource monitoring
    └─▶ Performance analysis
```

### Build Flow
```
User interaction
    │
    ▼
Layer palette → Canvas
    │
    ▼
Layer configuration
    │
    ▼
Model definition (internal)
    │
    ├─▶ Shape propagation
    ├─▶ Visualization
    └─▶ Code generation
    │
    ▼
Download/Export
    │
    ├─▶ .neural file
    ├─▶ TensorFlow code
    ├─▶ PyTorch code
    └─▶ ONNX code
```

## Configuration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Sources                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─▶ Default values (in code)
         │       Priority: 1 (lowest)
         │
         ├─▶ neural/config/features.yaml
         │       Priority: 2
         │
         ├─▶ Environment variables (NEURAL_FEATURE_*)
         │       Priority: 3
         │
         └─▶ Command-line arguments (--features)
                 Priority: 4 (highest)
         │
         ▼
    Merged Configuration
         │
         ├─▶ Feature Registry
         ├─▶ Server Settings
         └─▶ Service Activation
```

## Deployment Scenarios

### Local Development
```
Developer Machine
    │
    ▼
neural server start
    │
    ├─▶ Port 8050: Unified interface
    └─▶ All features enabled
        (Debug, Build, Monitor)
```

### Production Monitoring
```
Production Server
    │
    ▼
neural server start --features monitoring
    │
    └─▶ Port 8050: Monitor tab only
        (Drift detection, alerting, SLOs)
```

### CI/CD Pipeline
```
CI Server
    │
    ▼
neural compile model.neural --backend tensorflow
    │
    ├─▶ Validate DSL
    ├─▶ Generate code
    └─▶ Run tests
```

## Technology Stack

### Core
- **Python**: 3.8+
- **Lark**: DSL parsing
- **Click**: CLI framework
- **NumPy**: Numerical operations

### Web Interface
- **Dash**: Web framework
- **Plotly**: Visualizations
- **Flask**: HTTP server
- **Bootstrap**: UI components

### ML Backends (Optional)
- **TensorFlow**: Deep learning
- **PyTorch**: Deep learning
- **ONNX**: Model exchange

### Extended Features
- **Optuna**: HPO
- **scikit-learn**: ML utilities
- **Ray/Dask**: Distributed computing

### Experimental Features
- **WebSockets**: Real-time communication
- **FastAPI**: REST API

## Port Allocation Summary

| Port | Service              | Status      | Command                          |
|------|---------------------|-------------|----------------------------------|
| 8050 | Unified Interface   | ✅ Active   | `neural server start`            |
| 8000 | REST API (optional) | ⚠️ Optional | Auto-started if needed           |
| 8080 | WebSocket (optional)| ⚠️ Optional | Enable via feature flag          |
| 8051 | No-Code (legacy)    | ❌ Deprecated| Use unified server instead      |
| 8052 | Monitoring (legacy) | ❌ Deprecated| Use unified server instead      |
| 5001 | Internal WebSocket  | ❌ Deprecated| Replaced by unified server      |

## Future Architecture (v0.5.0+)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural DSL (Core)                            │
│  • Parser • Code Gen • Shape Propagation • CLI • HPO • AutoML  │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ neural-dsl   │      │ neural-      │      │ neural-      │
│ (unified     │      │ aquarium     │      │ marketplace  │
│  server)     │      │ (IDE)        │      │ (sharing)    │
└──────────────┘      └──────────────┘      └──────────────┘

        Modular architecture with separate packages
        for complex features that can be composed as needed
```

## See Also

- [SCOPE_CONSOLIDATION.md](./SCOPE_CONSOLIDATION.md) - Consolidation strategy
- [SERVICE_REGISTRY.md](./SERVICE_REGISTRY.md) - Service details
- [UNIFIED_SERVER_QUICKSTART.md](./UNIFIED_SERVER_QUICKSTART.md) - Quick start
- [AGENTS.md](./AGENTS.md) - Development guide
