"""
Neural Aquarium Source Components
"""

# Create plugins directory marker
import os
_plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
os.makedirs(_plugins_dir, exist_ok=True)
