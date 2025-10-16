import pytest

from neural.code_generation.code_generator import generate_code


def _simple_4d_input_output_model():
    return {
        "input": {"shape": (None, 8, 8, 3)},
        "layers": [
            {"type": "Output", "params": {"units": 10}},
        ],
        "loss": "mse",
        "optimizer": "Adam",
    }


def _conv_then_output_model():
    return {
        "input": {"shape": (None, 8, 8, 3)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 4, "kernel_size": 3}},
            {"type": "Output", "params": {"units": 5}},
        ],
        "loss": "mse",
        "optimizer": "Adam",
    }


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
@pytest.mark.parametrize("auto_flatten", [False, True])
def test_output_requires_flatten_policy(backend, auto_flatten):
    """Dense/Output on 4D input: strict policy vs auto-flatten.

    - When auto_flatten_output=True: code should include a flatten step before Dense/Output.
    - When False: generate_code should raise a ValueError explaining the policy.
    """
    if not auto_flatten:
        with pytest.raises(ValueError):
            generate_code(_simple_4d_input_output_model(), backend, auto_flatten_output=False)
        return

    code = generate_code(_simple_4d_input_output_model(), backend, auto_flatten_output=True)

    if backend == "tensorflow":
        has_flatten = "Flatten()" in code or "layers.Flatten()" in code
    else:
        has_flatten = ".view(" in code or ".reshape(" in code

    assert has_flatten, f"Expected flatten step for {backend} when auto_flatten_output=True"


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
def test_policy_parity_between_backends(backend):
    """Both backends should insert a flatten step before Dense/Output when policy allows it.

    This checks parity of the policy behavior across TF and PT by verifying the
    presence of a flatten step in both generated codes with auto_flatten_output=True.
    """
    code = generate_code(_conv_then_output_model(), backend, auto_flatten_output=True)

    if backend == "tensorflow":
        assert "Flatten()" in code or "layers.Flatten()" in code
        assert "Dense(" in code or "Output(" in code
    else:
        assert ".view(" in code or ".reshape(" in code
        assert "nn.Linear(" in code or "Output(" in code

