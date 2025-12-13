#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "================================"
echo "Neural Aquarium Build Script"
echo "================================"
echo ""

show_help() {
    cat << EOF
Usage: ./build/build.sh [OPTIONS]

Build Neural Aquarium for desktop platforms.

Options:
    -p, --platform PLATFORM    Target platform (win, mac, linux, all)
    -a, --arch ARCH           Target architecture (x64, arm64, universal, all)
    -s, --sign                Enable code signing
    -r, --release             Build for release (includes auto-update)
    -c, --clean               Clean build artifacts before building
    -h, --help                Show this help message

Examples:
    ./build/build.sh -p linux -a x64
    ./build/build.sh -p all -s -r
    ./build/build.sh -p mac -a universal --sign
    ./build/build.sh --clean -p win

Environment Variables:
    GH_TOKEN                  GitHub token for release publishing
    WIN_CSC_LINK             Windows certificate file path
    WIN_CSC_KEY_PASSWORD     Windows certificate password
    APPLE_ID                 Apple ID for notarization
    APPLE_ID_PASSWORD        App-specific password
    APPLE_TEAM_ID            Apple Team ID
    CSC_LINK                 macOS certificate (P12)
    CSC_KEY_PASSWORD         macOS certificate password
    GPG_PASSPHRASE           GPG passphrase for Linux signing

EOF
}

PLATFORM=""
ARCH=""
SIGN=false
RELEASE=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -a|--arch)
            ARCH="$2"
            shift 2
            ;;
        -s|--sign)
            SIGN=true
            shift
            ;;
        -r|--release)
            RELEASE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$PLATFORM" ]; then
    PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$PLATFORM" in
        darwin*)
            PLATFORM="mac"
            ;;
        linux*)
            PLATFORM="linux"
            ;;
        mingw*|msys*|cygwin*)
            PLATFORM="win"
            ;;
    esac
    echo "Auto-detected platform: $PLATFORM"
fi

if [ "$CLEAN" = true ]; then
    echo "Cleaning build artifacts..."
    rm -rf dist build node_modules/.cache
    echo "Clean complete."
    echo ""
fi

echo "Installing dependencies..."
npm ci

echo ""
echo "Building React application..."
npm run build

echo ""
echo "Building Electron application for $PLATFORM..."

BUILD_CMD="npm run electron:build"

case "$PLATFORM" in
    win|windows)
        BUILD_CMD="$BUILD_CMD:win"
        ;;
    mac|macos|darwin)
        BUILD_CMD="$BUILD_CMD:mac"
        ;;
    linux)
        BUILD_CMD="$BUILD_CMD:linux"
        ;;
    all)
        BUILD_CMD="$BUILD_CMD:all"
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        echo "Valid platforms: win, mac, linux, all"
        exit 1
        ;;
esac

if [ "$RELEASE" = true ]; then
    if [ -z "$GH_TOKEN" ]; then
        echo "Warning: GH_TOKEN not set. Publishing will fail."
        echo "Set GH_TOKEN environment variable for release builds."
    fi
    export PUBLISH="always"
else
    export PUBLISH="never"
fi

if [ "$SIGN" = false ]; then
    export CSC_IDENTITY_AUTO_DISCOVERY=false
fi

eval "$BUILD_CMD"

echo ""
echo "================================"
echo "Build complete!"
echo "================================"
echo ""
echo "Output directory: $PROJECT_DIR/dist"
echo ""

if [ -d "$PROJECT_DIR/dist" ]; then
    echo "Built files:"
    ls -lh "$PROJECT_DIR/dist" | grep -v "^d" | awk '{print "  " $9 " (" $5 ")"}'
    
    if [ -f "$PROJECT_DIR/dist/checksums-"* ]; then
        echo ""
        echo "Checksums generated:"
        cat "$PROJECT_DIR/dist/checksums-"* 2>/dev/null || true
    fi
fi

echo ""
echo "To test the application:"
echo "  npm run electron"
echo ""
echo "To create a release:"
echo "  git tag aquarium-v0.3.0"
echo "  git push origin aquarium-v0.3.0"
echo ""
