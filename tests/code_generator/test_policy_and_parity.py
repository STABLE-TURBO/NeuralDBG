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
    """Dense/Output on 4D input should auto-insert flatten only when flag is enabled.

    - When auto_flatten_output=True: code should include an explicit flatten step
      (Flatten() for TF, view(…,-1) for PT) before applying Dense/Output.
    - When False: ensure that no automatic flatten marker appears.
    """
    code = generate_code(_simple_4d_input_output_model(), backend, auto_flatten_output=auto_flatten)

    if backend == "tensorflow":
        has_flatten = "Flatten()" in code or "layers.Flatten()" in code
    else:
        has_flatten = ".view(" in code or ".reshape(" in code

    if auto_flatten:
        assert has_flatten, f"Expected flatten step for {backend} when auto_flatten_output=True"
    else:
        assert not has_flatten, f"Did not expect auto flatten for {backend} when auto_flatten_output=False"


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

