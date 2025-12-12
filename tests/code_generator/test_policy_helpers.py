import pytest

from neural.code_generation.shape_policy_helpers import (
    ensure_2d_before_dense_tf,
    ensure_2d_before_dense_pt,
)
from neural.shape_propagation.shape_propagator import ShapePropagator


def test_tf_helper_inserts_flatten_when_allowed():
    propagator = ShapePropagator(debug=False)
    shape = (None, 8, 8, 3)
    insert, new_shape = ensure_2d_before_dense_tf(
        rank_non_batch=3,
        auto_flatten_output=True,
        propagator=propagator,
        current_input_shape=shape,
    )
    assert "Flatten()" in insert
    assert new_shape is not None


def test_tf_helper_raises_when_not_allowed():
    propagator = ShapePropagator(debug=False)
    with pytest.raises(ValueError):
        ensure_2d_before_dense_tf(3, False, propagator, (None, 8, 8, 3))


def test_pt_helper_appends_view_when_allowed():
    propagator = ShapePropagator(debug=False)
    body = []
    shape = (None, 8, 8, 3)
    new_shape = ensure_2d_before_dense_pt(
        rank_non_batch=3,
        auto_flatten_output=True,
        forward_code_body=body,
        propagator=propagator,
        current_input_shape=shape,
    )
    assert any(".view(" in line for line in body)
    assert new_shape is not None

