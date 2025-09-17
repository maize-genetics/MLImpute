#!/bin/bash

# Main packaging script for "MLImpute" Tauri app
# This script orchestrates the complete packaging process including:
# 1. Download Python runtime for target platform
# 2. Build Python wheelhouse
# 3. Build Tauri application with bundled resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_MODE="release"
TARGET_PLATFORM="current"
GPU_SUPPORT="false"
SKIP_PYTHON_DOWNLOAD="false"
SKIP_WHEELHOUSE="false"

# Functions to print colored outputs
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode MODE           Build mode: 'debug' or 'release' (default: release)"
    echo "  --platform PLATFORM   Target platform: 'current', 'macos-arm64', 'macos-x64', 'linux-x64' (default: current)"
    echo "  --gpu                 Include GPU support (Linux and Windows)"
    echo "  --skip-python         Skip Python runtime download"
    echo "  --skip-wheelhouse     Skip wheelhouse build"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Package for current platform, release mode"
    echo "  $0 --mode debug                       # Package in debug mode"
    echo "  $0 --platform macos-arm64             # Package for macOS ARM64"
    echo "  $0 --gpu                              # Package with GPU support (Linux/Windows)"
    echo "  $0 --skip-python --skip-wheelhouse    # Skip Python setup, use existing"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        --platform)
            TARGET_PLATFORM="$2"
            shift 2
            ;;
        --gpu)
            GPU_SUPPORT="true"
            shift
            ;;
        --skip-python)
            SKIP_PYTHON_DOWNLOAD="true"
            shift
            ;;
        --skip-wheelhouse)
            SKIP_WHEELHOUSE="true"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate build mode
if [[ "$BUILD_MODE" != "debug" && "$BUILD_MODE" != "release" ]]; then
    print_error "Invalid build mode: $BUILD_MODE. Must be 'debug' or 'release'"
    exit 1
fi

print_status "Starting ML Impute packaging..."
print_status "Build mode: $BUILD_MODE"
print_status "Target platform: $TARGET_PLATFORM"
print_status "GPU support: $GPU_SUPPORT"

cd "$PROJECT_ROOT"

# Step 1: Download Python runtime (if not skipped)
if [[ "$SKIP_PYTHON_DOWNLOAD" != "true" ]]; then
    print_status "Downloading Python runtime..."
    if ! ./scripts/download-python-runtime.sh; then
        print_error "Failed to download Python runtime"
        exit 1
    fi
else
    print_warning "Skipping Python runtime download"
fi

# Step 2: Build wheelhouse (if not skipped)
if [[ "$SKIP_WHEELHOUSE" != "true" ]]; then
    print_status "Building Python wheelhouse..."
    if [[ "$GPU_SUPPORT" == "true" ]]; then
        if ! ./scripts/build-wheelhouse.sh --gpu; then
            print_error "Failed to build wheelhouse with GPU support"
            exit 1
        fi
    else
        if ! ./scripts/build-wheelhouse.sh; then
            print_error "Failed to build wheelhouse"
            exit 1
        fi
    fi
else
    print_warning "Skipping wheelhouse build"
fi

# Step 3: Install Node.js dependencies
print_status "Installing Node.js dependencies..."
if ! npm install; then
    print_error "Failed to install Node.js dependencies"
    exit 1
fi

# Step 4: Build frontend
print_status "Building frontend..."
if ! npm run build; then
    print_error "Failed to build frontend"
    exit 1
fi

# Step 5: Build Tauri application
print_status "Building Tauri application..."
cd src-tauri

if [[ "$BUILD_MODE" == "debug" ]]; then
    if ! cargo tauri build --debug; then
        print_error "Failed to build Tauri application in debug mode"
        exit 1
    fi
else
    if ! cargo tauri build; then
        print_error "Failed to build Tauri application in release mode"
        exit 1
    fi
fi

cd "$PROJECT_ROOT"

# Step 6: Show build results
print_status "Build completed successfully!"
print_status "Artifacts can be found in: src-tauri/target/$BUILD_MODE/"

# List the generated files
if [[ "$BUILD_MODE" == "release" ]]; then
    BUNDLE_DIR="src-tauri/target/release/bundle"
else
    BUNDLE_DIR="src-tauri/target/debug/bundle"
fi

if [[ -d "$BUNDLE_DIR" ]]; then
    print_status "Generated bundles:"
    find "$BUNDLE_DIR" -name "*.dmg" -o -name "*.app" -o -name "*.deb" -o -name "*.AppImage" -o -name "*.exe" -o -name "*.msi" | while read -r file; do
        echo "  - $file"
        if [[ -f "$file" ]]; then
            echo "    Size: $(du -h "$file" | cut -f1)"
        fi
    done
fi

print_status "Packaging complete!"
