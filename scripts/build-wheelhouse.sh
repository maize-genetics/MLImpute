#!/bin/bash

# Build wheelhouse script for packaging Python dependencies
# This script builds wheels for all dependencies that will be bundled with the app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELHOUSE_DIR="$PROJECT_ROOT/src-tauri/resources/wheelhouse"

echo "Building wheelhouse for platform: $(uname -m)-$(uname -s)"

# Create wheelhouse directory
mkdir -p "$WHEELHOUSE_DIR"

# Clean existing wheels
rm -rf "$WHEELHOUSE_DIR"/*

echo "Using pixi to build wheels..."

# Check if we're in a pixi environment, if not, use pixi run
if command -v pixi &> /dev/null && [[ -z "$PIXI_IN_SHELL" ]]; then
    cd "$PROJECT_ROOT"
    
    # Build CPU-only wheels by default
    echo "Building CPU-only wheels..."
    pixi run python -m pip wheel --wheel-dir "$WHEELHOUSE_DIR" -r requirements.txt
    
    # Optionally build GPU wheels if requested
    if [[ "$1" == "--gpu" ]] && ([[ "$(uname -s)" == "Linux" ]] || [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == CYGWIN* ]] || [[ "$(uname -s)" == MSYS* ]]); then
        echo "Building GPU-specific wheels..."
        pixi run --environment gpu python -m pip wheel --wheel-dir "$WHEELHOUSE_DIR" -r requirements-gpu.txt
    fi
else
    echo "ERROR: pixi not found or not in pixi environment"
    exit 1
fi

echo "Wheelhouse built successfully at: $WHEELHOUSE_DIR"

# Remove unnecessary files to reduce size
echo "Cleaning up wheelhouse to reduce size..."
find "$WHEELHOUSE_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
find "$WHEELHOUSE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$WHEELHOUSE_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$WHEELHOUSE_DIR" -name "*.pyo" -delete 2>/dev/null || true

echo "Wheels created:"
ls -la "$WHEELHOUSE_DIR"
echo "Total wheelhouse size: $(du -sh "$WHEELHOUSE_DIR" | cut -f1)"