# Neural DSL - Project Focus & Scope

## Core Mission

Neural DSL exists to solve **one problem exceptionally well**: making neural network prototyping faster and less error-prone through a declarative DSL with compile-time shape validation.

## What We Do Best

### 1. DSL-First Design
- Clean, readable syntax for network definition
- Declarative approach reduces boilerplate by 70-80%
- Framework-agnostic architecture descriptions

### 2. Shape Validation
- Compile-time dimension checking
- Prevents 90%+ of runtime shape errors
- Visual shape propagation diagrams

### 3. Multi-Backend Code Generation
- TensorFlow, PyTorch, and ONNX support
- Readable, idiomatic generated code
- Easy framework switching

### 4. Integrated Debugging
- NeuralDbg real-time dashboard
- Gradient flow visualization
- Memory and performance profiling

## Scope Boundaries

### ✅ In Scope (Core Features)

**DSL & Parser**
- Network definition language
- Layer composition and macros
- Input/output specifications
- Training configuration

**Shape Propagation**
- Static shape inference
- Dimension validation
- Type checking (basic)
- Error reporting

**Code Generation**
- TensorFlow generator
- PyTorch generator
- ONNX export
- Deployment optimization

**Debugging & Visualization**
- Architecture visualization
- Shape flow diagrams
- NeuralDbg dashboard
- Basic profiling

**CLI Interface**
- Compile, run, debug commands
- Visualization export
- Simple deployment

### ⚠️ Limited Scope (Supported but Not Focus)

**Hyperparameter Optimization**
- Basic Optuna integration
- Grid/random search only
- No advanced multi-fidelity methods

**Experiment Tracking**
- Local SQLite tracking
- Basic metric logging
- Simple comparison tools
- No distributed coordination

**Cloud Integrations**
- AWS SageMaker (basic)
- GCP Vertex AI (basic)
- Azure ML (basic)
- No complex workflows

**AutoML**
- Simple architecture search
- Limited search strategies
- Educational examples only
- Not production-grade NAS

### ❌ Out of Scope (Explicitly Excluded)

**IDE Development**
- Full-featured code editors
- Custom UI frameworks
- → Use existing editors (VS Code, PyCharm) with language server

**LLM/Chat Interfaces**
- Conversational model generation
- Natural language queries
- → Focus on clear DSL syntax instead

**Collaboration Tools**
- Real-time code sharing
- Team workspaces
- Version control systems
- → Use Git and existing tools

**Marketplace/Model Hub**
- Model sharing platform
- Community contributions
- → Use HuggingFace Hub integration

**Federated Learning**
- Distributed privacy-preserving training
- Secure aggregation
- → Too specialized, extract to separate project

**Production MLOps**
- CI/CD pipelines
- A/B testing frameworks
- Model monitoring
- → Use existing MLOps platforms

**Data Management**
- Data versioning systems
- Feature stores
- Data quality validation
- → Use DVC, Great Expectations, etc.

**Advanced Deployment**
- Kubernetes orchestration
- Multi-region distribution
- Auto-scaling infrastructure
- → Use platform-specific tools

## Feature Decision Framework

When evaluating new features, ask:

1. **Does it improve the core DSL experience?**
   - ✅ Better shape inference → YES
   - ❌ Built-in Git UI → NO

2. **Is it essential for neural network prototyping?**
   - ✅ More layer types → YES
   - ❌ Team chat system → NO

3. **Can it be better solved by existing tools?**
   - ✅ Version control → Use Git
   - ✅ Data versioning → Use DVC
   - ❌ Shape validation → No good alternative

4. **Does it require <500 LOC to maintain?**
   - ✅ Simple features → Consider
   - ❌ Complex subsystems → Extract or integrate

5. **Will 80%+ of users benefit?**
   - ✅ Better error messages → YES
   - ❌ Federated learning → <10% use case

## Integration Philosophy

**Prefer Integration Over Implementation**

Instead of building features, integrate with best-in-class tools:

- **Experiment Tracking**: MLflow, Weights & Biases, TensorBoard
- **Data Versioning**: DVC, LakeFS
- **Model Registry**: HuggingFace Hub, MLflow Registry
- **Deployment**: Platform-native tools (SageMaker, Vertex AI, AzureML)
- **Monitoring**: Prometheus, Grafana, custom platform tools
- **IDE**: VS Code, PyCharm with language server protocol

Provide **simple adapters**, not full reimplementations.

## User Personas

### Primary: Research Engineer / Student
- Prototyping architectures rapidly
- Learning neural network concepts
- Comparing different approaches
- Needs: Fast iteration, clear errors, good visualization

### Secondary: ML Engineer (Early Stage)
- Proof-of-concept development
- Framework evaluation
- Quick experiments
- Needs: Multi-backend support, easy export

### Non-Target: Production ML Team
- Large-scale deployment
- Complex MLOps pipelines
- Enterprise integrations
- Reason: Better served by specialized tools

## Success Metrics

### Quality Indicators
- **Error Prevention**: % of shape errors caught at compile time
- **Boilerplate Reduction**: Lines of DSL vs. equivalent framework code
- **Framework Parity**: Feature coverage across backends
- **Type Safety**: % of core code with type hints

### Usage Indicators
- **Time to First Model**: Minutes from install to running model
- **Documentation Clarity**: Can user achieve goal without asking?
- **Learning Curve**: Can newcomer understand DSL in <30 minutes?

### Anti-Metrics (What We Don't Optimize For)
- ❌ Total feature count
- ❌ Lines of code in repo
- ❌ Number of integrations
- ❌ UI polish

## Roadmap Principles

1. **Depth over Breadth**: Better core features, not more features
2. **Stability over Novelty**: Reliable fundamentals, not experimental add-ons
3. **Simplicity over Flexibility**: Cover 80% use cases perfectly
4. **Integration over Implementation**: Connect, don't duplicate
5. **Documentation over Features**: Users need to understand existing features

## Current State (v0.3.0)

**Strengths**
- Core DSL is solid and expressive
- Shape propagation works reliably
- Multi-backend code generation is functional
- NeuralDbg provides real value

**Weaknesses**
- Too many peripheral features dilute focus
- Type safety incomplete
- Documentation scattered
- Unclear scope boundaries

**Post-Cleanup Focus**
1. Deprecate out-of-scope features
2. Harden core with full type coverage
3. Improve error messages
4. Simplify installation and setup
5. Clear documentation structure

## Communication

When discussing features, use this vocabulary:

- **Core**: Essential to mission, actively maintained
- **Supported**: Works but minimal maintenance
- **Experimental**: May change or be removed
- **Deprecated**: Will be removed in future version
- **Out of Scope**: Won't be added, use integration instead

This keeps expectations clear and prevents scope creep.

## Conclusion

Neural DSL is **not** trying to be:
- An IDE
- An MLOps platform
- A data pipeline tool
- A collaboration suite

Neural DSL **is**:
- A powerful DSL for neural networks
- A shape validation system
- A multi-backend code generator
- A debugging and visualization tool

Everything else should integrate with existing solutions or be extracted to separate projects.

**Stay focused. Ship quality. Solve the core problem exceptionally well.**
