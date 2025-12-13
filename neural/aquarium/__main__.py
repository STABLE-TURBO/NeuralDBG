#!/usr/bin/env python
"""
Entry point for running Neural Aquarium as a module
Usage: python -m neural.aquarium
"""

import sys
from neural.aquarium.aquarium import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAquarium stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        sys.exit(1)
