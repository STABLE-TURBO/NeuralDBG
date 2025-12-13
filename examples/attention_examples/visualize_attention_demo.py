"""
Demonstration of attention visualization capabilities with Neural DSL.

This script shows how to programmatically use the AttentionVisualizer
to analyze transformer models compiled from Neural DSL.
"""

import numpy as np
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.explainability.attention_visualizer import AttentionVisualizer


def demo_basic_usage():
    """Demonstrate basic attention visualization from a DSL file."""
    print("="*70)
    print("Demo 1: Basic Attention Visualization")
    print("="*70)
    
    # Path to Neural DSL file
    dsl_file = Path(__file__).parent / "basic_transformer.neural"
    
    # Generate synthetic input data
    batch_size = 1
    seq_length = 50
    embed_dim = 256
    input_data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    print(f"Input shape: {input_data.shape}")
    print(f"DSL file: {dsl_file}")
    
    # Create visualizer
    visualizer = AttentionVisualizer(model=None, backend='tensorflow')
    
    # Visualize attention from DSL
    try:
        results = visualizer.visualize_from_dsl(
            dsl_file=str(dsl_file),
            input_data=input_data,
            backend='tensorflow',
            output_dir='demo_outputs_basic'
        )
        
        print(f"\nExtracted attention from {len(results['attention_weights'])} layers")
        
        for layer_name, weights in results['attention_weights'].items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {weights.shape}")
            
            # Analyze attention patterns
            analysis = visualizer.analyze_attention_patterns(weights)
            print(f"  Number of heads: {analysis['num_heads']}")
            print(f"  Average entropy: {analysis['avg_entropy']:.3f}")
            print(f"  Average max attention: {analysis['avg_max_attention']:.3f}")
            print(f"  Diagonal attention strength: {analysis['avg_diagonal_strength']:.3f}")
            
            # Per-head analysis
            print(f"  Per-head entropy: {[f'{e:.3f}' for e in analysis['attention_entropy']]}")
        
        print(f"\nOutputs saved to: demo_outputs_basic/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_tokens():
    """Demonstrate attention visualization with token labels."""
    print("\n" + "="*70)
    print("Demo 2: Attention Visualization with Token Labels")
    print("="*70)
    
    dsl_file = Path(__file__).parent / "basic_transformer.neural"
    
    # Generate input
    batch_size = 1
    seq_length = 10  # Use shorter sequence for clearer visualization
    embed_dim = 256
    input_data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    # Define token labels
    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "around", "."]
    
    print(f"Tokens: {tokens}")
    print(f"Input shape: {input_data.shape}")
    
    # Adjust model input if needed (create smaller version)
    # For this demo, we'll use the existing model but only visualize subset
    
    visualizer = AttentionVisualizer(model=None, backend='tensorflow')
    
    try:
        # Note: In production, you'd modify the input_shape in the DSL file
        # For demo purposes, we show how tokens would be used
        print("\nToken labels will be used in the visualization heatmaps")
        print("Each cell (i,j) shows how much token i attends to token j")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_multi_layer_analysis():
    """Demonstrate analyzing attention across multiple transformer layers."""
    print("\n" + "="*70)
    print("Demo 3: Multi-Layer Transformer Attention Analysis")
    print("="*70)
    
    dsl_file = Path(__file__).parent / "multi_layer_transformer.neural"
    
    # Generate input
    batch_size = 1
    seq_length = 128
    embed_dim = 512
    input_data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    print(f"Input shape: {input_data.shape}")
    print(f"DSL file: {dsl_file}")
    print(f"This model has 3 stacked transformer layers")
    
    visualizer = AttentionVisualizer(model=None, backend='tensorflow')
    
    try:
        results = visualizer.visualize_from_dsl(
            dsl_file=str(dsl_file),
            input_data=input_data,
            backend='tensorflow',
            output_dir='demo_outputs_multilayer'
        )
        
        print(f"\nFound {len(results['attention_weights'])} attention layers")
        
        # Compare attention patterns across layers
        print("\nLayer-by-layer analysis:")
        for layer_idx, (layer_name, weights) in enumerate(results['attention_weights'].items()):
            analysis = visualizer.analyze_attention_patterns(weights)
            
            print(f"\nLayer {layer_idx} ({layer_name}):")
            print(f"  Entropy: {analysis['avg_entropy']:.3f}")
            print(f"  Max attention: {analysis['avg_max_attention']:.3f}")
            print(f"  Diagonal strength: {analysis['avg_diagonal_strength']:.3f}")
            
            # Typically, later layers have lower entropy (more focused attention)
            if layer_idx == 0:
                print(f"  → First layer: Broad attention, learning basic patterns")
            elif layer_idx == len(results['attention_weights']) - 1:
                print(f"  → Last layer: Focused attention, making final decisions")
            else:
                print(f"  → Middle layer: Refining patterns")
        
        print(f"\nOutputs saved to: demo_outputs_multilayer/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_attention_head_comparison():
    """Demonstrate comparing different attention heads."""
    print("\n" + "="*70)
    print("Demo 4: Attention Head Comparison")
    print("="*70)
    
    print("Different attention heads learn different patterns:")
    print("  - Some heads focus on nearby tokens (local attention)")
    print("  - Some heads focus on distant tokens (long-range dependencies)")
    print("  - Some heads focus on specific syntactic/semantic relationships")
    
    # Generate sample attention weights for demonstration
    num_heads = 4
    seq_length = 20
    
    print(f"\nSimulating {num_heads} attention heads with {seq_length} tokens")
    
    # Create synthetic attention patterns
    attention_weights = np.zeros((num_heads, seq_length, seq_length))
    
    # Head 0: Local attention (diagonal pattern)
    for i in range(seq_length):
        for j in range(max(0, i-2), min(seq_length, i+3)):
            attention_weights[0, i, j] = np.exp(-abs(i-j))
    
    # Head 1: Global attention (uniform)
    attention_weights[1, :, :] = 1.0
    
    # Head 2: Attend to beginning (first few tokens)
    attention_weights[2, :, :5] = 2.0
    
    # Head 3: Attend to end (last few tokens)
    attention_weights[3, :, -5:] = 2.0
    
    # Normalize
    for h in range(num_heads):
        attention_weights[h] = attention_weights[h] / attention_weights[h].sum(axis=-1, keepdims=True)
    
    visualizer = AttentionVisualizer(model=None, backend='tensorflow')
    
    # Analyze each head
    for head_idx in range(num_heads):
        head_weights = attention_weights[head_idx:head_idx+1]
        analysis = visualizer.analyze_attention_patterns(head_weights)
        
        print(f"\nHead {head_idx}:")
        print(f"  Entropy: {analysis['attention_entropy'][0]:.3f}")
        print(f"  Diagonal strength: {analysis['diagonal_attention_strength'][0]:.3f}")
        
        if head_idx == 0:
            print(f"  → Type: Local attention (focuses on nearby positions)")
        elif head_idx == 1:
            print(f"  → Type: Global attention (attends to all positions)")
        elif head_idx == 2:
            print(f"  → Type: Prefix attention (focuses on start of sequence)")
        else:
            print(f"  → Type: Suffix attention (focuses on end of sequence)")


def demo_interactive_visualization():
    """Demonstrate creating interactive visualizations."""
    print("\n" + "="*70)
    print("Demo 5: Interactive Visualization")
    print("="*70)
    
    print("The AttentionVisualizer can create interactive HTML visualizations")
    print("using Plotly, allowing you to:")
    print("  - Zoom in/out on specific attention patterns")
    print("  - Hover to see exact attention weights")
    print("  - Compare multiple layers side-by-side")
    
    # Create sample attention data
    num_layers = 2
    num_heads = 4
    seq_length = 20
    
    attention_data = {}
    for layer_idx in range(num_layers):
        weights = np.random.rand(1, num_heads, seq_length, seq_length)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attention_data[f'transformer_encoder_{layer_idx}'] = weights
    
    visualizer = AttentionVisualizer(model=None, backend='tensorflow')
    
    try:
        output_path = visualizer.create_interactive_visualization(
            attention_data,
            tokens=[f'T{i}' for i in range(seq_length)],
            output_path='demo_interactive.html'
        )
        
        if output_path:
            print(f"\nInteractive visualization created: {output_path}")
            print("Open this file in a web browser to explore attention patterns")
        else:
            print("\nPlotly not available for interactive visualization")
            print("Install with: pip install plotly")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Neural DSL Attention Visualization Demo")
    print("="*70)
    
    # Run all demos
    demo_basic_usage()
    demo_with_tokens()
    demo_multi_layer_analysis()
    demo_attention_head_comparison()
    demo_interactive_visualization()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)
    print("\nTo visualize attention via CLI:")
    print("  neural visualize <model.neural> --attention")
    print("\nFor more options:")
    print("  neural visualize --help")
