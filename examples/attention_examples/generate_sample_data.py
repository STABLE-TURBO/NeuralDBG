"""
Generate sample input data for attention visualization examples.

This script creates synthetic input data that can be used to test
attention visualization with the Neural DSL transformer models.
"""

import numpy as np
import os


def generate_basic_transformer_data(output_dir='data'):
    """Generate data for basic_transformer.neural"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shape: (batch_size, sequence_length, embedding_dim)
    # basic_transformer.neural expects (50, 256)
    batch_size = 1
    seq_length = 50
    embed_dim = 256
    
    data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    output_path = os.path.join(output_dir, 'basic_transformer_input.npy')
    np.save(output_path, data)
    print(f"Generated basic transformer input data: {output_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    
    return output_path


def generate_multi_layer_transformer_data(output_dir='data'):
    """Generate data for multi_layer_transformer.neural"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shape: (batch_size, sequence_length, embedding_dim)
    # multi_layer_transformer.neural expects (128, 512)
    batch_size = 1
    seq_length = 128
    embed_dim = 512
    
    data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    output_path = os.path.join(output_dir, 'multi_layer_transformer_input.npy')
    np.save(output_path, data)
    print(f"Generated multi-layer transformer input data: {output_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    
    return output_path


def generate_text_classification_data(output_dir='data'):
    """Generate data for text_classification.neural"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shape: (batch_size, sequence_length, embedding_dim)
    # text_classification.neural has Embedding layer, so we can provide embedded data
    batch_size = 1
    seq_length = 200
    embed_dim = 256
    
    data = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float32)
    
    output_path = os.path.join(output_dir, 'text_classification_input.npy')
    np.save(output_path, data)
    print(f"Generated text classification input data: {output_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    
    return output_path


def generate_token_ids_data(output_dir='data', vocab_size=10000, seq_length=200):
    """Generate token ID data (for models with Embedding layer)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shape: (batch_size, sequence_length)
    batch_size = 1
    
    # Random token IDs
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)
    
    output_path = os.path.join(output_dir, 'token_ids_input.npy')
    np.save(output_path, token_ids)
    print(f"Generated token IDs data: {output_path}")
    print(f"  Shape: {token_ids.shape}")
    print(f"  Vocab size: {vocab_size}, Range: [{token_ids.min()}, {token_ids.max()}]")
    
    return output_path


def generate_realistic_attention_pattern(seq_length=50):
    """Generate more realistic synthetic input that creates interpretable attention patterns"""
    
    # Create embeddings with some structure
    embed_dim = 256
    data = np.zeros((1, seq_length, embed_dim), dtype=np.float32)
    
    # Add some periodic patterns to create interesting attention
    for i in range(seq_length):
        # Base random embedding
        data[0, i] = np.random.randn(embed_dim) * 0.1
        
        # Add periodic patterns (simulates repeated words/concepts)
        if i % 10 == 0:  # Every 10th position has similar pattern
            data[0, i] += np.random.randn(embed_dim) * 0.5
        
        # Add position encoding-like pattern
        for j in range(embed_dim):
            if j % 2 == 0:
                data[0, i, j] += np.sin(i / 10000 ** (j / embed_dim))
            else:
                data[0, i, j] += np.cos(i / 10000 ** (j / embed_dim))
    
    return data


def generate_structured_data(output_dir='data'):
    """Generate structured data that will produce more interpretable attention patterns"""
    os.makedirs(output_dir, exist_ok=True)
    
    data = generate_realistic_attention_pattern(seq_length=50)
    
    output_path = os.path.join(output_dir, 'structured_input.npy')
    np.save(output_path, data)
    print(f"Generated structured input data: {output_path}")
    print(f"  Shape: {data.shape}")
    print(f"  This data contains periodic patterns for more interpretable attention")
    
    return output_path


if __name__ == '__main__':
    print("Generating sample data for attention visualization examples...\n")
    
    # Generate data for each example
    generate_basic_transformer_data()
    print()
    
    generate_multi_layer_transformer_data()
    print()
    
    generate_text_classification_data()
    print()
    
    generate_token_ids_data()
    print()
    
    generate_structured_data()
    print()
    
    print("All sample data generated successfully!")
    print("\nUsage examples:")
    print("  neural visualize examples/attention_examples/basic_transformer.neural --attention --data data/basic_transformer_input.npy")
    print("  neural visualize examples/attention_examples/multi_layer_transformer.neural --attention --data data/multi_layer_transformer_input.npy")
    print("  neural visualize examples/attention_examples/text_classification.neural --attention --data data/text_classification_input.npy")
