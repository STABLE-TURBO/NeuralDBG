from __future__ import annotations

from typing import Any, Callable, Dict, List

def my_custom_layer(items: List[Any]) -> Dict[str, Any]:
    return {"type": "CustomLayer", "custom_param": int(items[0])}

def register(register_fn: Callable[[str, Callable[..., Any]], None]) -> None:
    register_fn("CustomLayer", my_custom_layer)
