#!/bin/bash

# Download python-build-standalone runtime for embedding
# This script downloads the appropriate Python runtime for the target platform

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/src-tauri/resources/python-runtime"

# Python Build Standalone version to use
PYTHON_VERSION="3.10.18"
PBS_VERSION="20250902"

# Detect platform
case "$(uname -s)" in
    Darwin)
        case "$(uname -m)" in
            arm64) PLATFORM="aarch64-apple-darwin" ;;
            x86_64) PLATFORM="x86_64-apple-darwin" ;;
            *) echo "Unsupported macOS architecture: $(uname -m)"; exit 1 ;;
        esac
        ;;
    Linux)
        case "$(uname -m)" in
            x86_64) PLATFORM="x86_64-unknown-linux-gnu" ;;
            *) echo "Unsupported Linux architecture: $(uname -m)"; exit 1 ;;
        esac
        ;;
    MINGW64_NT-*)
        # Windows with Git Bash/MINGW64
        case "$(uname -m)" in
            x86_64) PLATFORM="x86_64-pc-windows-msvc" ;;
            *) echo "Unsupported Windows architecture: $(uname -m)"; exit 1 ;;
        esac
        ;;
    CYGWIN_NT-*)
        # Windows with Cygwin
        case "$(uname -m)" in
            x86_64) PLATFORM="x86_64-pc-windows-msvc" ;;
            *) echo "Unsupported Windows architecture: $(uname -m)"; exit 1 ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $(uname -s)"
        exit 1
        ;;
esac

ARCHIVE_NAME="cpython-${PYTHON_VERSION}+${PBS_VERSION}-${PLATFORM}-install_only.tar.gz"
# TODO - vvv try to find a more stable release than this (this was suggested by ChatGPT...) vvv
DOWNLOAD_URL="https://github.com/indygreg/python-build-standalone/releases/download/${PBS_VERSION}/${ARCHIVE_NAME}"

echo "Downloading Python runtime for platform: $PLATFORM"
echo "Download URL: $DOWNLOAD_URL"

# Create runtime directory
mkdir -p "$RUNTIME_DIR"

# Clean existing runtime
rm -rf "$RUNTIME_DIR"/*

# Download and extract
cd "$RUNTIME_DIR"
if command -v curl &> /dev/null; then
    curl -L -o "$ARCHIVE_NAME" "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget -O "$ARCHIVE_NAME" "$DOWNLOAD_URL"
else
    echo "ERROR: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Extract
tar -xzf "$ARCHIVE_NAME"
rm "$ARCHIVE_NAME"

# The archive extracts to a 'python' directory
if [[ -d "python" ]]; then
    mv python/* .
    rmdir python
fi

echo "Python runtime downloaded and extracted to: $RUNTIME_DIR"

# Set Python executable path based on platform
case "$PLATFORM" in
    *windows*)
        PYTHON_EXE="$RUNTIME_DIR/python.exe"
        echo "Python executable: $PYTHON_EXE"
        ;;
    *)
        PYTHON_EXE="$RUNTIME_DIR/bin/python3"
        echo "Python executable: $PYTHON_EXE"
        ;;
esac

# Clean up runtime to reduce size
echo "Cleaning up Python runtime to reduce size..."
# Remove unnecessary files
find "$RUNTIME_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "*.pyo" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$RUNTIME_DIR" -name "test" -type d -exec rm -rf {} + 2>/dev/null || true
find "$RUNTIME_DIR" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
# Remove documentation and examples
find "$RUNTIME_DIR" -name "*.md" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "*.txt" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "*.rst" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "README*" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "LICENSE*" -delete 2>/dev/null || true
find "$RUNTIME_DIR" -name "CHANGELOG*" -delete 2>/dev/null || true

# Show size after cleanup
echo "Runtime size after cleanup: $(du -sh "$RUNTIME_DIR" | cut -f1)"

# Verify the runtime works
if [[ -x "$PYTHON_EXE" ]]; then
    echo "Python version:"
    "$PYTHON_EXE" --version
    echo "Python runtime is ready!"
else
    echo "ERROR: Python executable not found or not executable at: $PYTHON_EXE"
    exit 1
fi
