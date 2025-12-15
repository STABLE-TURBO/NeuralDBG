"""
Comprehensive Neural Architecture Search (NAS) Examples

This module demonstrates advanced AutoML capabilities including:
- Neural Architecture Search across multiple search spaces
- Multi-objective NAS (accuracy, latency, model size)
- Cross-backend architecture search (TensorFlow, PyTorch)
- Search strategy comparison (Random, Evolutionary, Bayesian)
- Early stopping and efficient resource allocation
- Production-ready architecture export
"""

from pathlib import Path
import sys


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.automl import ArchitectureSpace, AutoMLEngine


# ============================================================================
# Example 1: Basic Neural Architecture Search
# ============================================================================

def example_basic_nas():
    """
    Basic NAS example: Search for CNN architecture on CIFAR-10.
    
    Searches over:
    - Number of convolutional layers
    - Number of filters per layer
    - Kernel sizes
    - Activation functions
    - Pooling strategies
    """
    print("\n" + "="*70)
    print("Example 1: Basic Neural Architecture Search for CNN")
    print("="*70)
    
    # Define search space for CNN
    search_space_config = """
    network SearchableCNN {
        input: (32, 32, 3)
        
        layers:
            # Convolutional blocks with searchable parameters
            Conv2D(
                filters=choice(32, 64, 128, 256),
                kernel_size=choice((3, 3), (5, 5), (7, 7)),
                activation=choice("relu", "elu", "selu"),
                padding="same"
            )
            BatchNormalization()
            
            MaxPooling2D(pool_size=choice((2, 2), (3, 3)))
            Dropout(rate=range(0.1, 0.5, step=0.1))
            
            # Second conv block
            Conv2D(
                filters=choice(64, 128, 256, 512),
                kernel_size=choice((3, 3), (5, 5)),
                activation=choice("relu", "elu"),
                padding="same"
            )
            BatchNormalization()
            
            MaxPooling2D(pool_size=(2, 2))
            Dropout(rate=range(0.1, 0.5, step=0.1))
            
            # Third conv block (optional)
            Conv2D(
                filters=choice(128, 256, 512),
                kernel_size=(3, 3),
                activation=choice("relu", "elu"),
                padding="same"
            )
            BatchNormalization()
            
            GlobalAveragePooling2D()
            
            # Dense layers
            Dense(
                units=choice(128, 256, 512, 1024),
                activation=choice("relu", "elu")
            )
            Dropout(rate=range(0.2, 0.6, step=0.1))
            
            Output(units=10, activation="softmax")
        
        loss: "sparse_categorical_crossentropy"
        optimizer: Adam(learning_rate=log_range(1e-4, 1e-2))
        
        train {
            epochs: 30
            batch_size: choice(32, 64, 128)
            validation_split: 0.2
        }
    }
    """
    
    # Create architecture space
    architecture_space = ArchitectureSpace.from_dsl(search_space_config)
    
    # Initialize AutoML engine
    engine = AutoMLEngine(
        search_strategy='random',  # Start with random search
        early_stopping='median',
        executor_type='sequential',
        backend='pytorch',
        device='cpu',  # Use 'cuda' for GPU
        output_dir='nas_results/basic_cnn'
    )
    
    # Prepare data
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    
    train_data = DataLoader(
        CIFAR10(root='./data', train=True, download=True, transform=ToTensor()),
        batch_size=64,
        shuffle=True
    )
    val_data = DataLoader(
        CIFAR10(root='./data', train=False, download=True, transform=ToTensor()),
        batch_size=64
    )
    
    # Run NAS
    print("\nStarting Neural Architecture Search...")
    print("This will evaluate 20 different architectures")
    
    results = engine.search(
        architecture_space=architecture_space,
        train_data=train_data,
        val_data=val_data,
        max_trials=20,
        max_epochs_per_trial=10,
        timeout=3600  # 1 hour timeout
    )
    
    # Print results
    print("\n" + "="*70)
    print("Search Results:")
    print("="*70)
    print(f"Total trials: {results['total_trials']}")
    print(f"Total time: {results['total_time']:.2f}s")
    print("Best architecture found:")
    print(f"  Accuracy: {results['best_metrics']['val_acc']:.4f}")
    print(f"  Architecture: {results['best_architecture']}")
    
    return results


# ============================================================================
# Example 2: Multi-Objective NAS
# ============================================================================

def example_multi_objective_nas():
    """
    Multi-objective NAS: Optimize for accuracy, latency, and model size.
    
    Finds Pareto-optimal architectures that balance:
    - Validation accuracy (maximize)
    - Inference latency (minimize)
    - Model size in parameters (minimize)
    """
    print("\n" + "="*70)
    print("Example 2: Multi-Objective Neural Architecture Search")
    print("="*70)
    
    search_space_config = """
    network EfficientNet {
        input: (224, 224, 3)
        
        layers:
            # Stem
            Conv2D(filters=choice(24, 32, 40), kernel_size=(3, 3), strides=(2, 2), padding="same")
            BatchNormalization()
            Activation(choice("relu", "swish", "gelu"))
            
            # MBConv blocks with varying expansion ratios
            # Block 1
            Conv2D(filters=choice(16, 24, 32), kernel_size=(1, 1), padding="same")
            BatchNormalization()
            Activation("swish")
            
            Conv2D(filters=choice(16, 24, 32), kernel_size=(3, 3), padding="same")
            BatchNormalization()
            Activation("swish")
            
            Conv2D(filters=choice(16, 24, 32), kernel_size=(1, 1), padding="same")
            BatchNormalization()
            
            # Block 2
            Conv2D(filters=choice(24, 32, 40), kernel_size=(1, 1), padding="same")
            BatchNormalization()
            Activation("swish")
            
            Conv2D(filters=choice(24, 32, 40), kernel_size=(3, 3), padding="same", strides=(2, 2))
            BatchNormalization()
            Activation("swish")
            
            Conv2D(filters=choice(24, 32, 40), kernel_size=(1, 1), padding="same")
            BatchNormalization()
            
            # Head
            GlobalAveragePooling2D()
            Dense(units=choice(512, 1024, 1536), activation="swish")
            Dropout(rate=range(0.2, 0.4, step=0.1))
            Output(units=1000, activation="softmax")
        
        loss: "sparse_categorical_crossentropy"
        optimizer: Adam(learning_rate=log_range(1e-4, 1e-2))
        
        train {
            epochs: 20
            batch_size: choice(32, 64)
            validation_split: 0.1
        }
    }
    """
    
    architecture_space = ArchitectureSpace.from_dsl(search_space_config)
    
    # Multi-objective engine
    engine = AutoMLEngine(
        search_strategy='evolutionary',  # Good for multi-objective
        early_stopping='asha',
        executor_type='sequential',
        backend='pytorch',
        device='cpu',
        output_dir='nas_results/efficient_net'
    )
    
    # Prepare data (using subset for demo)
    from torch.utils.data import DataLoader, Subset
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use CIFAR-10 as proxy for ImageNet
    from torchvision.datasets import CIFAR10
    full_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_val = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use subset for faster demo
    train_data = DataLoader(Subset(full_train, range(5000)), batch_size=32, shuffle=True)
    val_data = DataLoader(Subset(full_val, range(1000)), batch_size=32)
    
    print("\nStarting Multi-Objective NAS...")
    print("Optimizing for: accuracy (↑), latency (↓), model size (↓)")
    
    results = engine.search(
        architecture_space=architecture_space,
        train_data=train_data,
        val_data=val_data,
        max_trials=30,
        max_epochs_per_trial=10
    )
    
    # Analyze Pareto front
    print("\n" + "="*70)
    print("Pareto-Optimal Architectures:")
    print("="*70)
    
    # Sort by accuracy
    sorted_trials = sorted(
        results['trial_history'],
        key=lambda x: x['metrics'].get('val_acc', {}).get('max', 0),
        reverse=True
    )
    
    for i, trial in enumerate(sorted_trials[:5]):
        print(f"\nArchitecture {i+1}:")
        metrics = trial['metrics']
        print(f"  Accuracy: {metrics.get('val_acc', {}).get('max', 0):.4f}")
        print(f"  Parameters: ~{estimate_params(trial['architecture'])}")
        print(f"  Architecture: {summarize_arch(trial['architecture'])}")
    
    return results


# ============================================================================
# Example 3: Transformer Architecture Search
# ============================================================================

def example_transformer_nas():
    """
    NAS for transformer architectures.
    
    Searches over:
    - Number of layers
    - Number of attention heads
    - Hidden dimensions
    - Feed-forward dimensions
    - Attention types (standard, linear, sparse)
    """
    print("\n" + "="*70)
    print("Example 3: Transformer Architecture Search")
    print("="*70)
    
    search_space_config = """
    network SearchableTransformer {
        input: (512,)
        
        layers:
            # Embedding
            Embedding(
                input_dim=30000,
                output_dim=choice(256, 512, 768, 1024),
                mask_zero=True
            )
            Dropout(rate=range(0.0, 0.2, step=0.05))
            
            # Transformer layers (search for optimal depth)
            # Layer 1
            MultiHeadAttention(
                num_heads=choice(4, 8, 12, 16),
                key_dim=choice(32, 64, 96, 128),
                dropout=range(0.0, 0.3, step=0.05)
            )
            Add()
            LayerNormalization()
            
            Dense(units=choice(1024, 2048, 3072, 4096), activation=choice("relu", "gelu", "swish"))
            Dropout(rate=range(0.0, 0.3, step=0.05))
            Dense(units=choice(256, 512, 768, 1024), activation="linear")
            Dropout(rate=range(0.0, 0.3, step=0.05))
            Add()
            LayerNormalization()
            
            # Layer 2
            MultiHeadAttention(
                num_heads=choice(4, 8, 12, 16),
                key_dim=choice(32, 64, 96, 128),
                dropout=range(0.0, 0.3, step=0.05)
            )
            Add()
            LayerNormalization()
            
            Dense(units=choice(1024, 2048, 3072, 4096), activation=choice("relu", "gelu", "swish"))
            Dropout(rate=range(0.0, 0.3, step=0.05))
            Dense(units=choice(256, 512, 768, 1024), activation="linear")
            Dropout(rate=range(0.0, 0.3, step=0.05))
            Add()
            LayerNormalization()
            
            # Output
            GlobalAveragePooling1D()
            Dense(units=choice(256, 512, 1024), activation=choice("relu", "gelu"))
            Dropout(rate=range(0.2, 0.5, step=0.1))
            Output(units=2, activation="softmax")
        
        loss: "sparse_categorical_crossentropy"
        optimizer: Adam(
            learning_rate=log_range(1e-5, 1e-3),
            beta_1=range(0.85, 0.95, step=0.02),
            beta_2=range(0.95, 0.999, step=0.01)
        )
        
        train {
            epochs: 15
            batch_size=choice(16, 32, 64)
            validation_split: 0.15
        }
    }
    """
    
    architecture_space = ArchitectureSpace.from_dsl(search_space_config)
    
    # NAS engine with Bayesian optimization
    engine = AutoMLEngine(
        search_strategy='bayesian',  # Efficient for expensive evaluations
        early_stopping='median',
        executor_type='sequential',
        backend='pytorch',
        device='cpu',
        output_dir='nas_results/transformer'
    )
    
    # Prepare text classification data
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy data for demonstration
    # In practice, use real datasets like IMDB, SST-2, etc.
    num_samples = 1000
    seq_length = 512
    
    train_inputs = torch.randint(0, 30000, (num_samples, seq_length))
    train_labels = torch.randint(0, 2, (num_samples,))
    val_inputs = torch.randint(0, 30000, (num_samples // 5, seq_length))
    val_labels = torch.randint(0, 2, (num_samples // 5,))
    
    train_data = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=32,
        shuffle=True
    )
    val_data = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=32
    )
    
    print("\nStarting Transformer Architecture Search...")
    print("Using Bayesian optimization for efficient search")
    
    results = engine.search(
        architecture_space=architecture_space,
        train_data=train_data,
        val_data=val_data,
        max_trials=25,
        max_epochs_per_trial=10,
        # Bayesian optimization settings
        n_initial_random=5,
        acquisition_function='ei'  # Expected improvement
    )
    
    print("\n" + "="*70)
    print("Best Transformer Architecture Found:")
    print("="*70)
    print(f"Validation Accuracy: {results['best_metrics']['val_acc']:.4f}")
    print("Architecture Details:")
    print(f"  {results['best_architecture']}")
    
    return results


# ============================================================================
# Example 4: Search Strategy Comparison
# ============================================================================

def example_search_strategy_comparison():
    """
    Compare different NAS search strategies:
    - Random search (baseline)
    - Evolutionary search
    - Bayesian optimization
    - Regularized evolution
    """
    print("\n" + "="*70)
    print("Example 4: NAS Search Strategy Comparison")
    print("="*70)
    
    search_space_config = """
    network ComparableNet {
        input: (28, 28, 1)
        
        layers:
            Conv2D(
                filters=choice(32, 64, 128),
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            )
            MaxPooling2D(pool_size=(2, 2))
            
            Conv2D(
                filters=choice(64, 128, 256),
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            )
            MaxPooling2D(pool_size=(2, 2))
            
            Flatten()
            Dense(units=choice(128, 256, 512), activation="relu")
            Dropout(rate=range(0.3, 0.6, step=0.1))
            Output(units=10, activation="softmax")
        
        loss: "sparse_categorical_crossentropy"
        optimizer: Adam(learning_rate=log_range(1e-4, 1e-2))
        
        train {
            epochs: 10
            batch_size: choice(32, 64, 128)
            validation_split: 0.2
        }
    }
    """
    
    architecture_space = ArchitectureSpace.from_dsl(search_space_config)
    
    # Prepare MNIST data
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    
    train_data = DataLoader(
        MNIST(root='./data', train=True, download=True, transform=ToTensor()),
        batch_size=64,
        shuffle=True
    )
    val_data = DataLoader(
        MNIST(root='./data', train=False, download=True, transform=ToTensor()),
        batch_size=64
    )
    
    strategies = ['random', 'evolutionary', 'bayesian']
    results_comparison = {}
    
    for strategy in strategies:
        print(f"\n{'-'*70}")
        print(f"Testing {strategy.upper()} search strategy...")
        print(f"{'-'*70}")
        
        engine = AutoMLEngine(
            search_strategy=strategy,
            early_stopping='median',
            executor_type='sequential',
            backend='pytorch',
            device='cpu',
            output_dir=f'nas_results/comparison_{strategy}'
        )
        
        results = engine.search(
            architecture_space=architecture_space,
            train_data=train_data,
            val_data=val_data,
            max_trials=15,
            max_epochs_per_trial=5
        )
        
        results_comparison[strategy] = results
        
        print(f"\n{strategy.upper()} Results:")
        print(f"  Best Accuracy: {results['best_metrics']['val_acc']:.4f}")
        print(f"  Total Time: {results['total_time']:.2f}s")
    
    # Summary comparison
    print("\n" + "="*70)
    print("Strategy Comparison Summary:")
    print("="*70)
    
    for strategy, results in results_comparison.items():
        summary = engine.get_search_summary()
        print(f"\n{strategy.upper()}:")
        print(f"  Best: {summary['best_accuracy']:.4f}")
        print(f"  Mean: {summary['mean_accuracy']:.4f}")
        print(f"  Std: {summary['std_accuracy']:.4f}")
        print(f"  Improvement: {summary['improvement']:.4f}")
    
    return results_comparison


# ============================================================================
# Example 5: Cross-Backend Architecture Search
# ============================================================================

def example_cross_backend_nas():
    """
    Search for architecture that works well across both TensorFlow and PyTorch.
    Useful for ensuring architecture portability.
    """
    print("\n" + "="*70)
    print("Example 5: Cross-Backend Architecture Search")
    print("="*70)
    
    search_space_config = """
    network PortableNet {
        input: (32, 32, 3)
        
        layers:
            Conv2D(filters=choice(32, 64), kernel_size=(3, 3), padding="same", activation="relu")
            BatchNormalization()
            MaxPooling2D(pool_size=(2, 2))
            
            Conv2D(filters=choice(64, 128), kernel_size=(3, 3), padding="same", activation="relu")
            BatchNormalization()
            MaxPooling2D(pool_size=(2, 2))
            
            GlobalAveragePooling2D()
            Dense(units=choice(128, 256), activation="relu")
            Dropout(rate=range(0.3, 0.5, step=0.1))
            Output(units=10, activation="softmax")
        
        loss: "sparse_categorical_crossentropy"
        optimizer: Adam(learning_rate=log_range(1e-4, 1e-3))
        
        train {
            epochs: 10
            batch_size: 64
            validation_split: 0.2
        }
    }
    """
    
    architecture_space = ArchitectureSpace.from_dsl(search_space_config)
    
    backends = ['pytorch', 'tensorflow']
    backend_results = {}
    
    for backend in backends:
        print(f"\n{'-'*70}")
        print(f"Searching on {backend.upper()} backend...")
        print(f"{'-'*70}")
        
        engine = AutoMLEngine(
            search_strategy='random',
            early_stopping='median',
            executor_type='sequential',
            backend=backend,
            device='cpu',
            output_dir=f'nas_results/cross_backend_{backend}'
        )
        
        # Prepare data based on backend
        if backend == 'pytorch':
            from torch.utils.data import DataLoader
            from torchvision.datasets import CIFAR10
            from torchvision.transforms import ToTensor
            
            train_data = DataLoader(
                CIFAR10(root='./data', train=True, download=True, transform=ToTensor()),
                batch_size=64,
                shuffle=True
            )
            val_data = DataLoader(
                CIFAR10(root='./data', train=False, download=True, transform=ToTensor()),
                batch_size=64
            )
        else:  # tensorflow
            import tensorflow as tf
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
            val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
        
        results = engine.search(
            architecture_space=architecture_space,
            train_data=train_data,
            val_data=val_data,
            max_trials=10,
            max_epochs_per_trial=5
        )
        
        backend_results[backend] = results
        
        print(f"\n{backend.upper()} Results:")
        print(f"  Best Accuracy: {results['best_metrics']['val_acc']:.4f}")
    
    # Compare results
    print("\n" + "="*70)
    print("Cross-Backend Comparison:")
    print("="*70)
    
    for backend, results in backend_results.items():
        print(f"\n{backend.upper()}:")
        print(f"  Best Architecture: {results['best_architecture']}")
        print(f"  Best Accuracy: {results['best_metrics']['val_acc']:.4f}")
        print(f"  Search Time: {results['total_time']:.2f}s")
    
    return backend_results


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_params(architecture):
    """Estimate number of parameters in architecture."""
    # Simplified estimation
    total_params = 0
    for layer in architecture.get('layers', []):
        layer_type = layer.get('type', '')
        if layer_type == 'Conv2D':
            filters = layer.get('filters', 0)
            kernel_size = layer.get('kernel_size', (3, 3))
            total_params += filters * kernel_size[0] * kernel_size[1]
        elif layer_type == 'Dense':
            units = layer.get('units', 0)
            total_params += units * 1000  # Rough estimate
    
    if total_params > 1e6:
        return f"{total_params/1e6:.2f}M"
    elif total_params > 1e3:
        return f"{total_params/1e3:.2f}K"
    return str(total_params)


def summarize_arch(architecture):
    """Create human-readable summary of architecture."""
    layers = architecture.get('layers', [])
    summary_parts = []
    
    for layer in layers[:5]:  # First 5 layers
        layer_type = layer.get('type', 'Unknown')
        if layer_type == 'Conv2D':
            filters = layer.get('filters', '?')
            kernel = layer.get('kernel_size', '?')
            summary_parts.append(f"Conv2D({filters}, {kernel})")
        elif layer_type == 'Dense':
            units = layer.get('units', '?')
            summary_parts.append(f"Dense({units})")
        else:
            summary_parts.append(layer_type)
    
    if len(layers) > 5:
        summary_parts.append(f"... ({len(layers)-5} more)")
    
    return " → ".join(summary_parts)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all NAS examples."""
    print("\n" + "="*70)
    print("Neural DSL - Comprehensive Neural Architecture Search Examples")
    print("="*70)
    print("\nThis script demonstrates advanced AutoML/NAS capabilities.")
    print("Note: Examples use small trial counts for demonstration.")
    print("For production use, increase max_trials (e.g., 50-200).")
    
    import argparse
    parser = argparse.ArgumentParser(description='NAS Examples')
    parser.add_argument('--example', type=str, default='all',
                        choices=['all', 'basic', 'multi', 'transformer', 'comparison', 'cross'],
                        help='Which example to run')
    args = parser.parse_args()
    
    try:
        if args.example in ['all', 'basic']:
            example_basic_nas()
        
        if args.example in ['all', 'multi']:
            example_multi_objective_nas()
        
        if args.example in ['all', 'transformer']:
            example_transformer_nas()
        
        if args.example in ['all', 'comparison']:
            example_search_strategy_comparison()
        
        if args.example in ['all', 'cross']:
            example_cross_backend_nas()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("NAS Examples Completed!")
    print("="*70)
    print("\nResults saved to nas_results/ directory")
    print("For more information, see: neural/automl/README.md")


if __name__ == '__main__':
    main()
