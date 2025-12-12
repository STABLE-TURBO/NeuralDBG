# Neural DSL Tutorials

Welcome to the Neural DSL tutorial collection! These resources will help you master Neural DSL from beginner to advanced levels.

## 📚 Tutorial Formats

We provide tutorials in multiple formats to suit your learning style:

### 🎓 Interactive Notebooks
Jupyter notebooks with executable code and visualizations:
- [Quickstart Tutorial](quickstart_tutorial.ipynb) - Your first Neural DSL model (15 min)
- [HPO Tutorial](hpo_tutorial.ipynb) - Hyperparameter optimization guide (30 min)

### 🎬 Video Tutorials
Step-by-step video walkthroughs (coming soon):
- Getting Started (5 min)
- Building Your First Model (10 min)
- Hyperparameter Optimization (8 min)
- Debugging with NeuralDbg (7 min)
- Multi-Backend Compilation (6 min)
- Cloud Integration (8 min)

See [video_scripts.md](video_scripts.md) for detailed scripts and production notes.

### 📖 Text Guides
Comprehensive written documentation:
- [Troubleshooting Guide](../troubleshooting.md) - Fix common issues
- [Migration Guide](../migration.md) - Upgrade between versions
- [DSL Documentation](../dsl.md) - Complete syntax reference
- [CLI Reference](../cli.md) - Command-line interface guide

## 🎯 Learning Paths

### Beginner Path
**Goal:** Create and train your first model

1. **[Quickstart Tutorial](quickstart_tutorial.ipynb)** ⭐ START HERE
   - Install Neural DSL
   - Create a simple model
   - Compile and train
   - Make predictions
   - **Time:** 15 minutes

2. **[MNIST Commented Example](../../examples/mnist_commented.neural)**
   - Understand layer-by-layer architecture
   - Learn shape propagation
   - Training configuration
   - **Time:** 20 minutes reading

3. **[Troubleshooting Guide](../troubleshooting.md)**
   - Common errors and fixes
   - Shape mismatch debugging
   - Installation issues
   - **Reference:** As needed

### Intermediate Path
**Goal:** Optimize models and use advanced features

1. **[HPO Tutorial](hpo_tutorial.ipynb)** ⭐ KEY SKILL
   - Optimize hyperparameters automatically
   - Learn HPO syntax
   - Multi-parameter optimization
   - **Time:** 30 minutes

2. **[Sentiment Analysis Example](../../examples/sentiment_analysis_commented.neural)**
   - RNN/LSTM architectures
   - Text preprocessing
   - Sequence modeling
   - **Time:** 25 minutes reading

3. **[Migration Guide](../migration.md)**
   - Version upgrades
   - Framework migration (TensorFlow ↔ PyTorch)
   - Best practices
   - **Reference:** As needed

### Advanced Path
**Goal:** Master all Neural DSL features

1. **NeuralDbg Debugging** (video tutorial)
   - Real-time monitoring
   - Gradient flow analysis
   - Dead neuron detection
   - Anomaly detection
   - **Time:** 7 minutes

2. **Multi-Backend Compilation** (video tutorial)
   - TensorFlow vs PyTorch
   - ONNX export
   - Framework switching
   - **Time:** 6 minutes

3. **Cloud Integration** (video tutorial)
   - Kaggle notebooks
   - Google Colab
   - AWS SageMaker
   - Remote debugging
   - **Time:** 8 minutes

## 📋 Tutorial Quick Reference

| Tutorial | Level | Time | Format | Topics |
|----------|-------|------|--------|--------|
| Quickstart | Beginner | 15 min | Notebook | Installation, First Model, Training |
| HPO | Intermediate | 30 min | Notebook | Optimization, HPO Syntax, Best Practices |
| MNIST Commented | Beginner | 20 min | DSL | CNN Architecture, Shape Flow |
| Sentiment Commented | Intermediate | 25 min | DSL | RNN/LSTM, Text Processing |
| Getting Started | Beginner | 5 min | Video | Quick Introduction |
| Building Models | Beginner | 10 min | Video | Architecture Design |
| HPO Video | Intermediate | 8 min | Video | Optimization |
| NeuralDbg | Advanced | 7 min | Video | Debugging Tools |
| Multi-Backend | Intermediate | 6 min | Video | Framework Switching |
| Cloud | Advanced | 8 min | Video | Cloud Platforms |

## 🎓 Complete Course Outline

### Week 1: Fundamentals
- **Day 1:** Installation and first model (Quickstart Tutorial)
- **Day 2:** Understanding architectures (MNIST Commented)
- **Day 3:** Training and evaluation
- **Day 4:** Visualization and shape propagation
- **Day 5:** Practice: Build your own classifier

### Week 2: Optimization
- **Day 1:** HPO introduction (HPO Tutorial)
- **Day 2:** Learning rate optimization
- **Day 3:** Architecture search
- **Day 4:** Multi-parameter optimization
- **Day 5:** Practice: Optimize a model

### Week 3: Advanced Topics
- **Day 1:** RNN/LSTM (Sentiment Analysis Commented)
- **Day 2:** Debugging with NeuralDbg
- **Day 3:** Multi-backend compilation
- **Day 4:** Cloud integration
- **Day 5:** Final project

## 🛠️ Hands-On Projects

Apply your knowledge with these projects:

### Project 1: Image Classifier (Beginner)
**Goal:** Build a classifier for your own dataset

**Steps:**
1. Prepare image dataset (cats vs dogs, flowers, etc.)
2. Create Neural DSL model based on MNIST example
3. Adjust input shape and output classes
4. Train and evaluate
5. Visualize results

**Skills:** Model creation, training, evaluation

### Project 2: Text Analyzer (Intermediate)
**Goal:** Sentiment analysis on movie reviews

**Steps:**
1. Download IMDB dataset
2. Preprocess text to sequences
3. Create LSTM model (use sentiment example)
4. Run HPO to optimize
5. Deploy best model

**Skills:** NLP, HPO, deployment

### Project 3: Multi-Model System (Advanced)
**Goal:** Compare TensorFlow and PyTorch

**Steps:**
1. Create complex architecture in DSL
2. Compile to both backends
3. Train and compare performance
4. Deploy better version
5. Document findings

**Skills:** Multi-backend, performance analysis

## 📊 Assessment Checklist

### Beginner Level ✓
- [ ] Can install Neural DSL
- [ ] Can write basic model definitions
- [ ] Understands layer types (Dense, Conv2D, etc.)
- [ ] Can compile and train models
- [ ] Can interpret training metrics
- [ ] Knows how to fix shape mismatches

### Intermediate Level ✓
- [ ] Can use HPO syntax
- [ ] Understands when to use different HPO functions
- [ ] Can optimize multiple parameters
- [ ] Knows how to debug training issues
- [ ] Can work with RNN/LSTM models
- [ ] Can migrate between versions

### Advanced Level ✓
- [ ] Masters NeuralDbg dashboard
- [ ] Can switch between backends effectively
- [ ] Can deploy to cloud platforms
- [ ] Can create custom macros
- [ ] Can optimize for production
- [ ] Can contribute to Neural DSL

## 🔗 Additional Resources

### Documentation
- [DSL Syntax Reference](../dsl.md) - Complete language specification
- [CLI Reference](../cli.md) - All command-line options
- [Installation Guide](../installation.md) - Detailed setup instructions
- [Troubleshooting Guide](../troubleshooting.md) - Problem solving
- [Migration Guide](../migration.md) - Version upgrades

### Examples
- [Basic Examples](../../examples/) - Simple models
- [HPO Examples](../../examples/mnist_hpo.neural) - Optimization demos
- [Advanced Architectures](../../examples/transformer.neural) - Complex models

### Community
- [Discord Server](https://discord.gg/KFku4KvS) - Get help, share projects
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions) - Q&A
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues) - Bug reports

### External Resources
- [TensorFlow Documentation](https://www.tensorflow.org/) - Backend reference
- [PyTorch Documentation](https://pytorch.org/) - Backend reference
- [Deep Learning Book](https://www.deeplearningbook.org/) - Theory background

## 💡 Learning Tips

### For Beginners
1. **Start with Quickstart** - Don't skip it!
2. **Read commented examples** - They explain WHY, not just WHAT
3. **Run all code** - Hands-on practice is essential
4. **Ask questions** - Use Discord when stuck
5. **Build small projects** - Apply knowledge immediately

### For Intermediate Users
1. **Focus on HPO** - It makes the biggest difference
2. **Understand your data** - Preprocessing is crucial
3. **Use NeuralDbg** - Catch problems early
4. **Experiment with backends** - Find what works best
5. **Read migration guide** - Stay up to date

### For Advanced Users
1. **Contribute examples** - Help the community
2. **Optimize for production** - Performance matters
3. **Explore cloud platforms** - Scale up
4. **Share your work** - Blog posts, tutorials
5. **Report bugs** - Make Neural DSL better

## 🎬 Creating Video Tutorials

Interested in creating video tutorials? Check out:
- [Video Scripts](video_scripts.md) - Complete scripts for all videos
- Production guidelines and recording setup
- Editing recommendations
- Publishing checklist

We welcome community-created tutorials! Contact us on Discord to coordinate.

## 🤝 Contributing Tutorials

Want to add a tutorial? We'd love your help!

**Tutorial Ideas:**
- Domain-specific guides (medical imaging, NLP, etc.)
- Performance optimization techniques
- Integration with other tools
- Real-world case studies
- Language translations

**Submission Process:**
1. Discuss idea on Discord or GitHub Discussions
2. Create tutorial following existing format
3. Test all code examples
4. Submit pull request
5. Address review feedback

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

## 📞 Get Help

**Stuck on a tutorial?**
1. Check the troubleshooting guide
2. Search GitHub Issues
3. Ask on Discord (#tutorials channel)
4. Create a new discussion on GitHub

**Found an error?**
1. Report on GitHub Issues
2. Tag with "documentation" label
3. Provide tutorial name and section

## 🎉 Next Steps

Ready to start learning? Here's what to do:

1. **Complete the Quickstart Tutorial** → [Start Now](quickstart_tutorial.ipynb)
2. **Join our Discord** → [Get Support](https://discord.gg/KFku4KvS)
3. **Star the repo** → [Show Support](https://github.com/Lemniscate-world/Neural)
4. **Share your progress** → Tweet with #NeuralDSL

Happy learning! 🚀
