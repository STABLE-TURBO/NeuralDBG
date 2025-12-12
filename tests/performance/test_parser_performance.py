"""
Benchmark parser performance with optimized grammar.
"""
import time
import pytest
from neural.parser.parser import create_parser


SIMPLE_NETWORK = """
network SimpleNet {
    input: (28, 28, 1)
    layers: 
        Conv2D(32, kernel_size=3)
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128)
        Output(10)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
"""

COMPLEX_NETWORK = """
network ComplexNet {
    input: (224, 224, 3)
    layers:
        Conv2D(64, kernel_size=3, padding="same")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(128, kernel_size=3, padding="same")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(256, kernel_size=3, padding="same")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(512)
        Dropout(0.5)
        Dense(256)
        Dropout(0.3)
        Output(1000)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001, momentum=0.9)
    train {
        epochs: 100
        batch_size: 32
    }
}
"""


def test_parser_creation_time():
    """Test parser instantiation time."""
    start = time.time()
    parser = create_parser()
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"Parser creation took {elapsed:.3f}s, expected < 1.0s"
    print(f"\n✓ Parser creation time: {elapsed:.3f}s")


def test_simple_network_parse_time():
    """Test parsing time for simple network."""
    parser = create_parser()
    
    start = time.time()
    tree = parser.parse(SIMPLE_NETWORK)
    elapsed = time.time() - start
    
    assert tree is not None
    assert elapsed < 0.1, f"Simple network parse took {elapsed:.3f}s, expected < 0.1s"
    print(f"✓ Simple network parse time: {elapsed:.3f}s")


def test_complex_network_parse_time():
    """Test parsing time for complex network."""
    parser = create_parser()
    
    start = time.time()
    tree = parser.parse(COMPLEX_NETWORK)
    elapsed = time.time() - start
    
    assert tree is not None
    assert elapsed < 0.2, f"Complex network parse took {elapsed:.3f}s, expected < 0.2s"
    print(f"✓ Complex network parse time: {elapsed:.3f}s")


def test_repeated_parsing():
    """Test parsing the same network multiple times (cache effectiveness)."""
    parser = create_parser()
    
    times = []
    for i in range(10):
        start = time.time()
        tree = parser.parse(SIMPLE_NETWORK)
        elapsed = time.time() - start
        times.append(elapsed)
        assert tree is not None
    
    avg_time = sum(times) / len(times)
    assert avg_time < 0.15, f"Average parse time {avg_time:.3f}s, expected < 0.15s"
    print(f"✓ Average parse time (10 iterations): {avg_time:.3f}s")


def test_transformer_parse_time():
    """Test parsing time for ModelTransformer."""
    from neural.parser.parser import ModelTransformer
    parser = create_parser()
    
    tree = parser.parse(SIMPLE_NETWORK)
    
    start = time.time()
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    elapsed = time.time() - start
    
    assert model_data is not None
    assert elapsed < 0.2, f"Transformation took {elapsed:.3f}s, expected < 0.2s"
    print(f"✓ Model transformation time: {elapsed:.3f}s")


def test_lalr_parser_efficiency():
    """Test that LALR parser is used (more efficient than Earley)."""
    parser = create_parser()
    
    assert hasattr(parser, 'options')
    assert parser.options.parser == 'lalr', "Parser should use LALR for better performance"
    print("✓ Using efficient LALR parser")


def test_lexer_configuration():
    """Test that lexer is properly optimized."""
    parser = create_parser()
    
    assert hasattr(parser, 'options')
    assert parser.options.lexer in ['basic', 'contextual'], "Lexer should be basic or contextual"
    print(f"✓ Using {parser.options.lexer} lexer")


def test_grammar_cache():
    """Test that parser uses grammar caching."""
    parser = create_parser()
    
    assert hasattr(parser, 'options')
    assert parser.options.cache is True, "Grammar caching should be enabled"
    print("✓ Grammar caching enabled")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
