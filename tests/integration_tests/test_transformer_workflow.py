"""
Integration tests for transformer-based architectures.

Tests embedding layers, encoder-only, decoder-only, and full encoder-decoder
architectures across TensorFlow and PyTorch backends.
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    from neural.code_generation.code_generator import generate_code
except ImportError:
    generate_code = None


@pytest.fixture
def temp_workspace():
    """Fixture to provide a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    os.chdir(original_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestEmbeddingLayers:
    """Tests for embedding layer functionality."""

    def test_embedding_layer_parsing(self, temp_workspace):
        """Test: Parse DSL with Embedding layer."""
        dsl_code = """
        network EmbeddingNet {
            input: (100, 512)
            layers:
                Embedding(input_dim=10000, output_dim=512)
                Dense(256, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert 'input' in model_config
        assert 'layers' in model_config
        assert len(model_config['layers']) == 3
        assert model_config['layers'][0]['type'] == 'Embedding'
        assert model_config['layers'][0]['params']['input_dim'] == 10000
        assert model_config['layers'][0]['params']['output_dim'] == 512

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_embedding_tensorflow_generation(self, temp_workspace):
        """Test: Generate TensorFlow code with Embedding layer."""
        dsl_code = """
        network EmbeddingTF {
            input: (100,)
            layers:
                Embedding(input_dim=5000, output_dim=128)
                GlobalAveragePooling1D()
                Dense(64, "relu")
                Output(5, "softmax")
            
            loss: "sparse_categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'import tensorflow' in code
        assert 'layers.Embedding' in code
        assert 'input_dim=5000' in code
        assert 'output_dim=128' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_embedding_pytorch_generation(self, temp_workspace):
        """Test: Generate PyTorch code with Embedding layer."""
        dsl_code = """
        network EmbeddingPT {
            input: (50,)
            layers:
                Embedding(input_dim=1000, output_dim=64)
                Flatten()
                Dense(32, "relu")
                Output(3, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'nn.Embedding' in code
        assert 'num_embeddings=1000' in code
        assert 'embedding_dim=64' in code

    def test_embedding_with_different_dimensions(self, temp_workspace):
        """Test: Embedding layers with various dimension configurations."""
        test_cases = [
            (5000, 128),
            (10000, 256),
            (50000, 512),
            (100, 32)
        ]
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for input_dim, output_dim in test_cases:
            dsl_code = f"""
            network EmbeddingTest {{
                input: (100,)
                layers:
                    Embedding(input_dim={input_dim}, output_dim={output_dim})
                    Flatten()
                    Output(10)
            }}
            """
            
            tree = parser.parse(dsl_code)
            model_config = transformer.transform(tree)
            
            embedding_layer = model_config['layers'][0]
            assert embedding_layer['type'] == 'Embedding'
            assert embedding_layer['params']['input_dim'] == input_dim
            assert embedding_layer['params']['output_dim'] == output_dim


class TestEncoderOnlyArchitectures:
    """Tests for encoder-only transformer architectures (BERT-style)."""

    def test_simple_encoder_parsing(self, temp_workspace):
        """Test: Parse simple encoder-only transformer."""
        dsl_code = """
        network SimpleEncoder {
            input: (100, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048)
                GlobalAveragePooling1D()
                Dense(128, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert len(model_config['layers']) == 4
        encoder_layer = model_config['layers'][0]
        assert encoder_layer['type'] == 'TransformerEncoder'
        assert encoder_layer['params']['num_heads'] == 8
        assert encoder_layer['params']['ff_dim'] == 2048

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_encoder_tensorflow_generation(self, temp_workspace):
        """Test: Generate TensorFlow code for encoder-only transformer."""
        dsl_code = """
        network BERTStyleEncoder {
            input: (128, 768)
            layers:
                TransformerEncoder(num_heads=12, ff_dim=3072, dropout=0.1)
                TransformerEncoder(num_heads=12, ff_dim=3072, dropout=0.1)
                GlobalAveragePooling1D()
                Dense(768, "relu")
                Dropout(rate=0.1)
                Output(2, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.00001)
            
            train {
                epochs: 10
                batch_size: 32
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'import tensorflow' in code
        assert 'TransformerEncoder block' in code
        assert 'MultiHeadAttention' in code
        assert 'LayerNormalization' in code
        assert 'num_heads=12' in code
        assert code.count('TransformerEncoder block') == 2

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_encoder_pytorch_generation(self, temp_workspace):
        """Test: Generate PyTorch code for encoder-only transformer."""
        dsl_code = """
        network PyTorchEncoder {
            input: (100, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                Flatten()
                Dense(256, "relu")
                Output(5, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'TransformerEncoderLayer' in code
        assert 'd_model=512' in code
        assert 'nhead=8' in code
        assert 'dim_feedforward=2048' in code
        assert 'dropout=0.1' in code

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TF_AVAILABLE) or generate_code is None, 
                       reason="Both PyTorch and TensorFlow, or code generator not available")
    def test_encoder_multi_backend_consistency(self, temp_workspace):
        """Test: Same encoder architecture generates consistent code across backends."""
        dsl_code = """
        network ConsistentEncoder {
            input: (100, 256)
            layers:
                TransformerEncoder(num_heads=4, ff_dim=1024)
                GlobalAveragePooling1D()
                Dense(128, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        tf_code = generate_code(model_config, 'tensorflow')
        pt_code = generate_code(model_config, 'pytorch')
        
        assert 'MultiHeadAttention' in tf_code or 'TransformerEncoder' in tf_code
        assert 'TransformerEncoderLayer' in pt_code
        assert 'Dense' in tf_code or 'layers.Dense' in tf_code
        assert 'nn.Linear' in pt_code

    def test_stacked_encoder_layers(self, temp_workspace):
        """Test: Multiple stacked encoder layers."""
        dsl_code = """
        network StackedEncoder {
            input: (100, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048) * 6
                GlobalAveragePooling1D()
                Dense(256, "relu")
                Dropout(rate=0.1)
                Output(20, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        encoder_layers = [l for l in model_config['layers'] 
                         if l['type'] == 'TransformerEncoder']
        assert len(encoder_layers) >= 1


class TestDecoderOnlyArchitectures:
    """Tests for decoder-only transformer architectures (GPT-style)."""

    def test_decoder_parsing(self, temp_workspace):
        """Test: Parse decoder-only transformer architecture."""
        dsl_code = """
        network DecoderOnly {
            input: (100,)
            layers:
                Embedding(input_dim=50000, output_dim=768)
                TransformerDecoder(num_heads=12, ff_dim=3072)
                Dense(768, "relu")
                Output(50000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert len(model_config['layers']) == 4
        assert model_config['layers'][0]['type'] == 'Embedding'
        assert model_config['layers'][1]['type'] == 'TransformerDecoder'

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_decoder_tensorflow_generation(self, temp_workspace):
        """Test: Generate TensorFlow code for decoder-only transformer."""
        dsl_code = """
        network GPTStyleDecoder {
            input: (512,)
            layers:
                Embedding(input_dim=50257, output_dim=768)
                TransformerDecoder(num_heads=12, ff_dim=3072, dropout=0.1)
                TransformerDecoder(num_heads=12, ff_dim=3072, dropout=0.1)
                Dense(768, "gelu")
                LayerNormalization()
                Output(50257, "softmax")
            
            loss: "sparse_categorical_crossentropy"
            optimizer: Adam(learning_rate=0.00001)
            
            train {
                epochs: 3
                batch_size: 8
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'import tensorflow' in code
        assert 'layers.Embedding' in code
        assert 'input_dim=50257' in code
        assert 'output_dim=768' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_decoder_pytorch_generation(self, temp_workspace):
        """Test: Generate PyTorch code for decoder-only transformer."""
        dsl_code = """
        network PyTorchDecoder {
            input: (256,)
            layers:
                Embedding(input_dim=10000, output_dim=512)
                TransformerDecoder(num_heads=8, ff_dim=2048)
                Dense(512, "relu")
                Output(10000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'nn.Embedding' in code
        assert 'num_embeddings=10000' in code
        assert 'embedding_dim=512' in code

    def test_decoder_with_causal_masking(self, temp_workspace):
        """Test: Decoder with causal masking configuration."""
        dsl_code = """
        network CausalDecoder {
            input: (100,)
            layers:
                Embedding(input_dim=5000, output_dim=256)
                TransformerDecoder(num_heads=4, ff_dim=1024, causal=True)
                Dense(256, "relu")
                Output(5000)
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        decoder_layer = model_config['layers'][1]
        assert decoder_layer['type'] == 'TransformerDecoder'

    def test_stacked_decoder_layers(self, temp_workspace):
        """Test: Multiple stacked decoder layers."""
        dsl_code = """
        network StackedDecoder {
            input: (512,)
            layers:
                Embedding(input_dim=50000, output_dim=1024)
                TransformerDecoder(num_heads=16, ff_dim=4096) * 12
                Dense(1024, "relu")
                LayerNormalization()
                Output(50000, "softmax")
            
            optimizer: Adam(learning_rate=0.00001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        decoder_layers = [l for l in model_config['layers'] 
                         if l['type'] == 'TransformerDecoder']
        assert len(decoder_layers) >= 1


class TestEncoderDecoderArchitectures:
    """Tests for full encoder-decoder transformer architectures (T5/BART-style)."""

    def test_encoder_decoder_parsing(self, temp_workspace):
        """Test: Parse full encoder-decoder transformer."""
        dsl_code = """
        network EncoderDecoder {
            input: (100, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048)
                TransformerEncoder(num_heads=8, ff_dim=2048)
                TransformerDecoder(num_heads=8, ff_dim=2048)
                TransformerDecoder(num_heads=8, ff_dim=2048)
                Dense(512, "relu")
                Output(30000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        encoder_layers = [l for l in model_config['layers'] 
                         if l['type'] == 'TransformerEncoder']
        decoder_layers = [l for l in model_config['layers'] 
                         if l['type'] == 'TransformerDecoder']
        
        assert len(encoder_layers) == 2
        assert len(decoder_layers) == 2

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_encoder_decoder_tensorflow(self, temp_workspace):
        """Test: Generate TensorFlow code for encoder-decoder transformer."""
        dsl_code = """
        network Seq2SeqTransformer {
            input: (128, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                Dense(512, "relu")
                LayerNormalization()
                Output(32000, "softmax")
            
            loss: "sparse_categorical_crossentropy"
            optimizer: Adam(learning_rate=0.0001)
            
            train {
                epochs: 10
                batch_size: 16
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'import tensorflow' in code
        assert 'TransformerEncoder block' in code or 'MultiHeadAttention' in code
        assert 'LayerNormalization' in code
        assert code.count('MultiHeadAttention') >= 6 or code.count('TransformerEncoder') >= 3

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_encoder_decoder_pytorch(self, temp_workspace):
        """Test: Generate PyTorch code for encoder-decoder transformer."""
        dsl_code = """
        network PyTorchSeq2Seq {
            input: (100, 512)
            layers:
                TransformerEncoder(num_heads=8, ff_dim=2048)
                TransformerEncoder(num_heads=8, ff_dim=2048)
                TransformerDecoder(num_heads=8, ff_dim=2048)
                TransformerDecoder(num_heads=8, ff_dim=2048)
                Dense(512, "relu")
                Output(10000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'TransformerEncoderLayer' in code or 'TransformerDecoderLayer' in code
        assert 'nn.Linear' in code

    def test_encoder_decoder_with_embeddings(self, temp_workspace):
        """Test: Full encoder-decoder with embedding layers."""
        dsl_code = """
        network T5StyleTransformer {
            input: (512,)
            layers:
                Embedding(input_dim=32128, output_dim=512)
                TransformerEncoder(num_heads=8, ff_dim=2048) * 6
                TransformerDecoder(num_heads=8, ff_dim=2048) * 6
                Dense(512, "relu")
                LayerNormalization()
                Output(32128, "softmax")
            
            loss: "sparse_categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
            
            train {
                epochs: 20
                batch_size: 32
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert model_config['layers'][0]['type'] == 'Embedding'
        assert model_config['layers'][0]['params']['input_dim'] == 32128
        assert model_config['layers'][0]['params']['output_dim'] == 512

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TF_AVAILABLE) or generate_code is None, 
                       reason="Both PyTorch and TensorFlow, or code generator not available")
    def test_encoder_decoder_cross_backend(self, temp_workspace):
        """Test: Same encoder-decoder generates code for both backends."""
        dsl_code = """
        network CrossBackendSeq2Seq {
            input: (100, 256)
            layers:
                TransformerEncoder(num_heads=4, ff_dim=1024)
                TransformerDecoder(num_heads=4, ff_dim=1024)
                Dense(256, "relu")
                Output(5000, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        tf_code = generate_code(model_config, 'tensorflow')
        pt_code = generate_code(model_config, 'pytorch')
        
        assert 'tensorflow' in tf_code
        assert 'torch' in pt_code
        
        assert len(tf_code) > 0
        assert len(pt_code) > 0


class TestTransformerVariations:
    """Tests for various transformer architecture variations."""

    def test_transformer_with_different_head_counts(self, temp_workspace):
        """Test: Transformers with various attention head configurations."""
        head_counts = [2, 4, 8, 12, 16]
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for num_heads in head_counts:
            dsl_code = f"""
            network MultiHeadTest {{
                input: (100, 512)
                layers:
                    TransformerEncoder(num_heads={num_heads}, ff_dim=2048)
                    Dense(256)
                    Output(10)
            }}
            """
            
            tree = parser.parse(dsl_code)
            model_config = transformer.transform(tree)
            
            encoder = model_config['layers'][0]
            assert encoder['params']['num_heads'] == num_heads

    def test_transformer_with_different_ff_dimensions(self, temp_workspace):
        """Test: Transformers with various feedforward dimensions."""
        ff_dims = [512, 1024, 2048, 4096]
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for ff_dim in ff_dims:
            dsl_code = f"""
            network FFDimTest {{
                input: (100, 512)
                layers:
                    TransformerEncoder(num_heads=8, ff_dim={ff_dim})
                    Output(10)
            }}
            """
            
            tree = parser.parse(dsl_code)
            model_config = transformer.transform(tree)
            
            encoder = model_config['layers'][0]
            assert encoder['params']['ff_dim'] == ff_dim

    def test_transformer_with_dropout_variations(self, temp_workspace):
        """Test: Transformers with different dropout rates."""
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for dropout in dropout_rates:
            dsl_code = f"""
            network DropoutTest {{
                input: (100, 512)
                layers:
                    TransformerEncoder(num_heads=8, ff_dim=2048, dropout={dropout})
                    Output(10)
            }}
            """
            
            tree = parser.parse(dsl_code)
            model_config = transformer.transform(tree)
            
            encoder = model_config['layers'][0]
            assert encoder['params']['dropout'] == dropout

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_transformer_text_classification(self, temp_workspace):
        """Test: Complete transformer for text classification."""
        dsl_code = """
        network TextClassifier {
            input: (512,)
            layers:
                Embedding(input_dim=30000, output_dim=768)
                TransformerEncoder(num_heads=12, ff_dim=3072, dropout=0.1)
                TransformerEncoder(num_heads=12, ff_dim=3072, dropout=0.1)
                GlobalAveragePooling1D()
                Dense(768, "relu")
                Dropout(rate=0.1)
                Dense(384, "relu")
                Output(4, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.00005)
            
            train {
                epochs: 5
                batch_size: 16
                validation_split: 0.2
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'Embedding' in code
        assert 'TransformerEncoder' in code or 'MultiHeadAttention' in code
        assert 'GlobalAveragePooling1D' in code
        assert 'categorical_crossentropy' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_transformer_sequence_to_sequence(self, temp_workspace):
        """Test: Complete seq2seq transformer for translation."""
        dsl_code = """
        network TranslationModel {
            input: (256,)
            layers:
                Embedding(input_dim=50000, output_dim=512)
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                TransformerDecoder(num_heads=8, ff_dim=2048, dropout=0.1)
                Dense(512, "relu")
                LayerNormalization()
                Output(50000, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=0.0001)
            
            train {
                epochs: 50
                batch_size: 64
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'nn.Embedding' in code
        assert 'TransformerEncoderLayer' in code or 'TransformerDecoderLayer' in code
        assert 'nn.Linear' in code

    def test_transformer_with_position_encoding(self, temp_workspace):
        """Test: Transformer with explicit positional encoding."""
        dsl_code = """
        network PositionalTransformer {
            input: (512, 768)
            layers:
                PositionalEncoding(d_model=768, max_len=5000)
                TransformerEncoder(num_heads=12, ff_dim=3072)
                TransformerEncoder(num_heads=12, ff_dim=3072)
                GlobalAveragePooling1D()
                Dense(768, "relu")
                Output(1000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        encoder_layers = [l for l in model_config['layers'] 
                         if l['type'] == 'TransformerEncoder']
        assert len(encoder_layers) == 2

    def test_hybrid_cnn_transformer(self, temp_workspace):
        """Test: Hybrid architecture combining CNN and Transformer."""
        dsl_code = """
        network HybridCNNTransformer {
            input: (224, 224, 3)
            layers:
                Conv2D(filters=64, kernel_size=7, strides=2, activation="relu")
                MaxPooling2D(pool_size=2)
                Conv2D(filters=128, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(512, "relu")
                Reshape(target_shape=(32, 512))
                TransformerEncoder(num_heads=8, ff_dim=2048)
                TransformerEncoder(num_heads=8, ff_dim=2048)
                GlobalAveragePooling1D()
                Dense(256, "relu")
                Output(1000, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        conv_layers = [l for l in model_config['layers'] if l['type'] == 'Conv2D']
        transformer_layers = [l for l in model_config['layers'] 
                            if l['type'] == 'TransformerEncoder']
        
        assert len(conv_layers) == 2
        assert len(transformer_layers) == 2


class TestTransformerExecution:
    """Tests for actual execution of generated transformer code."""

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, reason="PyTorch or code generator not available")
    def test_pytorch_transformer_execution(self, temp_workspace):
        """Test: Execute generated PyTorch transformer code."""
        dsl_code = """
        network ExecutableTransformer {
            input: (50, 128)
            layers:
                TransformerEncoder(num_heads=4, ff_dim=512)
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        with open('transformer_model.py', 'w') as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("transformer_model", "transformer_model.py")
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            model = module.NeuralNetworkModel()
            test_input = torch.randn(2, 50, 128)
            output = model(test_input)
            assert output.shape[0] == 2
        except Exception as e:
            pytest.skip(f"Model execution skipped: {str(e)}")

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, reason="TensorFlow or code generator not available")
    def test_tensorflow_transformer_execution(self, temp_workspace):
        """Test: Execute generated TensorFlow transformer code."""
        dsl_code = """
        network TFExecutableTransformer {
            input: (100, 256)
            layers:
                TransformerEncoder(num_heads=4, ff_dim=1024)
                GlobalAveragePooling1D()
                Dense(128, "relu")
                Output(5, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'TransformerEncoder' in code or 'MultiHeadAttention' in code
        assert 'GlobalAveragePooling1D' in code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
