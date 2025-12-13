#!/bin/bash

set -e

case "$1" in
    remove|purge)
        # Update desktop database
        if command -v update-desktop-database > /dev/null 2>&1; then
            update-desktop-database -q /usr/share/applications || true
        fi
        
        # Update mime database
        if command -v update-mime-database > /dev/null 2>&1; then
            update-mime-database /usr/share/mime || true
        fi
        
        # Update icon cache
        if command -v gtk-update-icon-cache > /dev/null 2>&1; then
            gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
        fi
        
        # Remove symlink
        if [ -L "/usr/bin/neural-aquarium" ]; then
            rm -f /usr/bin/neural-aquarium || true
        fi
        ;;
esac

exit 0
