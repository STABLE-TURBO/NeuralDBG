"""
Neural API module.
"""

def create_app():
    from .main import create_app as _create_app
    return _create_app()

__all__ = [
    'create_app',
]
