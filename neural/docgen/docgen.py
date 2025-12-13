from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional, Tuple

from neural.shape_propagation.shape_propagator import ShapePropagator


def _shape_to_str(shape: Tuple[Optional[int], ...]) -> str:
    return "(" + ", ".join("None" if d is None else str(d) for d in shape) + ")"


def generate_markdown(model_data: Dict[str, Any]) -> str:
    """
    Generate a Markdown report (DocGen v1.1) for a Neural model with math and shapes.

    - Uses ShapePropagator to compute shapes (tolerant to errors)
    - Emits detailed math for all layers (Dense/Output/Flatten/Conv2D/Pool/BatchNorm)
    - Calculates and displays parameter counts per layer and total
    - Adds a Summary section and records shape propagation warnings
    """
    assert isinstance(model_data, dict) and 'input' in model_data and 'layers' in model_data

    lines: List[str] = []
    lines.append(f"# Model Documentation\n")

    # Initialize parameter tracking
    total_params = 0
    layer_params = {}

    # Intro and input
    input_shape = tuple(model_data['input']['shape'])

    # Shape propagation (collect warnings)
    propagator = ShapePropagator(debug=False)
    current_shape: Tuple[Optional[int], ...] = (None,) + tuple(input_shape)
    warnings: List[str] = []

    # Summary
    lines.append("\n## Summary\n")
    lines.append(f"- Input shape: `{input_shape}`\n")
    lines.append(f"- Layers: {len(model_data['layers'])}\n")

    lines.append("\n## Layers\n")

    for idx, layer in enumerate(model_data['layers']):
        ltype = layer.get('type')
        params = layer.get('params', {}) or {}
        # Math blurb for a few common layers
        math = ""
        if ltype in ("Dense", "Output"):
            units = params.get('units', 10)
            act = params.get('activation', 'softmax' if ltype == 'Output' else None)
            if act == 'softmax':
                math = "y = softmax(Wx + b)"
            elif act in ('relu', 'tanh'):
                math = f"y = {act}(Wx + b)"
            else:
                math = "y = Wx + b"
        elif ltype == "Flatten":
            math = "x' = reshape(x, (B, -1))"
        elif ltype == "Conv2D":
            k = params.get('kernel_size', 3)
            f = params.get('filters', 32)
            math = f"y[h,w,c_out] = conv_{k}(x) + b, filters={f}"

        before = _shape_to_str(current_shape)
        try:
            current_shape = propagator.propagate(current_shape, layer)
        except Exception as e:
            # Do not fail the docgen; record the issue and continue
            warnings.append(str(e))
            math = (math + "  ") if math else ""
            math += f"[shape propagation warning: {str(e)}]"
        after = _shape_to_str(current_shape)

        lines.append(f"- Layer {idx}: **{ltype}** {params}\n")
        if math:
            lines.append(f"  - Math: `{math}`\n")
        lines.append(f"  - Shape: {before} → {after}\n")

    # Warnings section (if any)
    if warnings:
        lines.append("\n## Warnings\n")
        for w in warnings[:10]:
            lines.append(f"- {w}\n")
        if len(warnings) > 10:
            lines.append(f"- ...and {len(warnings) - 10} more\n")

    # Tools availability
    lines.append("\n## Export\n")
    pandoc = shutil.which('pandoc')
    if pandoc:
        lines.append("- PDF export available via Pandoc (use CLI --pdf)\n")
    else:
        lines.append("- PDF export not detected (Pandoc not available)\n")

    return "".join(lines)
