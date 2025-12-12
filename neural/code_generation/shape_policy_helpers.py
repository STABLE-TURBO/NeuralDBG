import logging
from neural.shape_propagation.shape_propagator import ShapePropagator
from typing import Tuple, List

logger = logging.getLogger(__name__)


def ensure_2d_before_dense_tf(
    rank_non_batch: int,
    auto_flatten_output: bool,
    propagator: ShapePropagator,
    current_input_shape: tuple,
) -> Tuple[str, tuple]:
    """Ensure 2D input for Dense/Output in TensorFlow.

    Returns (insert_code, updated_shape). insert_code is a string to append to the TF code.
    Raises ValueError when higher-rank input is not allowed and auto_flatten_output is False.
    """
    if rank_non_batch > 1:
        if auto_flatten_output:
            insert = "x = layers.Flatten()(x)\n"
            try:
                current_input_shape = propagator.propagate(current_input_shape, {"type": "Flatten"})
            except Exception as e:
                logger.warning(f"Shape propagation warning (auto-flatten): {e}")
            return insert, current_input_shape
        raise ValueError(
            "Layer 'Output' expects 2D input (batch, features) but got higher-rank. "
            "Insert a Flatten/GAP before it or pass auto_flatten_output=True."
        )
    return "", current_input_shape


def ensure_2d_before_dense_pt(
    rank_non_batch: int,
    auto_flatten_output: bool,
    forward_code_body: List[str],
    propagator: ShapePropagator,
    current_input_shape: tuple,
) -> tuple:
    """Ensure 2D input for Dense/Output in PyTorch.

    Mutates forward_code_body when flatten is inserted and returns the updated shape.
    Raises ValueError when higher-rank input is not allowed and auto_flatten_output is False.
    """
    if rank_non_batch > 1:
        if auto_flatten_output:
            forward_code_body.append("x = x.view(x.size(0), -1)  # Flatten input")
            try:
                current_input_shape = propagator.propagate(current_input_shape, {"type": "Flatten"})
            except Exception as e:
                logger.warning(f"Shape propagation warning (auto-flatten): {e}")
            return current_input_shape
        raise ValueError(
            "Layer 'Output' expects 2D input (batch, features) but got higher-rank. "
            "Insert a Flatten/GAP before it or pass auto_flatten_output=True."
        )
    return current_input_shape


def get_rank_non_batch(current_input_shape: tuple) -> int:
    """Calculate the rank excluding the batch dimension."""
    try:
        return max(0, len(current_input_shape) - 1)
    except Exception:
        return 0
