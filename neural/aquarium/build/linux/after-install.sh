#!/bin/bash

set -e

case "$1" in
    configure)
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
        
        # Update man database
        if command -v mandb > /dev/null 2>&1; then
            mandb -q || true
        fi
        
        # Create symlink in /usr/bin if needed
        if [ ! -L "/usr/bin/neural-aquarium" ]; then
            ln -sf /opt/Neural\ Aquarium/neural-aquarium /usr/bin/neural-aquarium || true
        fi
        ;;
esac

exit 0
