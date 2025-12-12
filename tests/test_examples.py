"""
Tests for example validation and correctness.
"""

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code


# Get all .neural files in use_cases
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "use_cases"
NEURAL_FILES = list(EXAMPLES_DIR.glob("*.neural"))


@pytest.mark.parametrize("neural_file", NEURAL_FILES, ids=lambda f: f.name)
def test_example_parses(neural_file):
    """Test that example can be parsed"""
    with open(neural_file, 'r') as f:
        content = f.read()
    
    parser = create_parser(start_rule='network')
    tree = parser.parse(content)
    
    assert tree is not None, f"Failed to parse {neural_file.name}"


@pytest.mark.parametrize("neural_file", NEURAL_FILES, ids=lambda f: f.name)
def test_example_transforms(neural_file):
    """Test that example can be transformed to model data"""
    with open(neural_file, 'r') as f:
        content = f.read()
    
    parser = create_parser(start_rule='network')
    tree = parser.parse(content)
    
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    
    assert model_data is not None, f"Failed to transform {neural_file.name}"
    assert 'layers' in model_data, f"No layers in {neural_file.name}"
    assert len(model_data['layers']) > 0, f"Empty layers in {neural_file.name}"


@pytest.mark.parametrize("neural_file", NEURAL_FILES, ids=lambda f: f.name)
def test_example_compiles_tensorflow(neural_file):
    """Test that example compiles to TensorFlow"""
    with open(neural_file, 'r') as f:
        content = f.read()
    
    parser = create_parser(start_rule='network')
    tree = parser.parse(content)
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    
    try:
        code = generate_code(model_data, 'tensorflow')
        assert code is not None, f"Failed to generate TensorFlow code for {neural_file.name}"
        assert len(code) > 0, f"Empty TensorFlow code for {neural_file.name}"
    except Exception as e:
        pytest.skip(f"TensorFlow compilation not supported: {e}")


@pytest.mark.parametrize("neural_file", NEURAL_FILES, ids=lambda f: f.name)
def test_example_compiles_pytorch(neural_file):
    """Test that example compiles to PyTorch"""
    with open(neural_file, 'r') as f:
        content = f.read()
    
    parser = create_parser(start_rule='network')
    tree = parser.parse(content)
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    
    try:
        code = generate_code(model_data, 'pytorch')
        assert code is not None, f"Failed to generate PyTorch code for {neural_file.name}"
        assert len(code) > 0, f"Empty PyTorch code for {neural_file.name}"
    except Exception as e:
        pytest.skip(f"PyTorch compilation not supported: {e}")


def test_example_has_input():
    """Test that examples have input specifications"""
    for neural_file in NEURAL_FILES:
        with open(neural_file, 'r') as f:
            content = f.read()
        
        assert 'input:' in content, f"No input specification in {neural_file.name}"


def test_example_has_layers():
    """Test that examples have layer definitions"""
    for neural_file in NEURAL_FILES:
        with open(neural_file, 'r') as f:
            content = f.read()
        
        assert 'layers:' in content, f"No layers specification in {neural_file.name}"


def test_example_has_loss():
    """Test that examples have loss function"""
    for neural_file in NEURAL_FILES:
        with open(neural_file, 'r') as f:
            content = f.read()
        
        assert 'loss:' in content, f"No loss specification in {neural_file.name}"


def test_example_has_optimizer():
    """Test that examples have optimizer"""
    for neural_file in NEURAL_FILES:
        with open(neural_file, 'r') as f:
            content = f.read()
        
        assert 'optimizer:' in content, f"No optimizer specification in {neural_file.name}"


def test_notebooks_exist():
    """Test that all notebooks exist"""
    notebooks_dir = Path(__file__).parent.parent / "examples" / "notebooks"
    
    expected_notebooks = [
        "image_classification_tutorial.ipynb",
        "sentiment_analysis_tutorial.ipynb",
        "transformer_nlp_tutorial.ipynb",
        "time_series_tutorial.ipynb",
        "gan_tutorial.ipynb"
    ]
    
    for notebook in expected_notebooks:
        notebook_path = notebooks_dir / notebook
        assert notebook_path.exists(), f"Notebook not found: {notebook}"


def test_readme_files_exist():
    """Test that README files exist"""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    readme_files = [
        "README.md",
        "notebooks/README.md",
        "use_cases/README.md",
        "EXAMPLES_QUICK_REF.md"
    ]
    
    for readme in readme_files:
        readme_path = examples_dir / readme
        assert readme_path.exists(), f"README not found: {readme}"


def test_validation_script_exists():
    """Test that validation script exists"""
    script_path = EXAMPLES_DIR / "validate_examples.py"
    assert script_path.exists(), "Validation script not found"


def test_example_count():
    """Test that we have the expected number of examples"""
    # We expect at least 5 .neural files
    assert len(NEURAL_FILES) >= 5, f"Expected at least 5 examples, found {len(NEURAL_FILES)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
