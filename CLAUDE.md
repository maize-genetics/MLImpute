# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLImpute is a machine learning-based haplotype imputation tool that combines multiple approaches including BiMamba (state space models), ModernBERT (transformer models), and HMM-based methods. The project consists of:

1. **Tauri Desktop Application** - A cross-platform GUI built with React frontend and Rust backend
2. **Python CLI Tool** - Command-line interface for running imputation models
3. **Machine Learning Pipeline** - Various ML models for genetic data imputation

## Development Commands

### Environment Setup
```bash
# Using pixi (recommended):
pixi install                     # CPU-only environment
pixi install --environment gpu   # GPU environment (Linux only)

# Using conda:
conda env create -f environment.yml
```

### Frontend Development (Tauri)
```bash
npm install                # Install Node.js dependencies
npm run dev               # Start Vite development server
npm run build             # Build frontend for production
npm run tauri dev         # Launch Tauri app in development mode
```

### Python CLI Usage
```bash
# With pixi:
pixi run -- python src/impute.py --input <file> --output <file> --model <model>

# With conda:
python src/impute.py --input <file> --output <file> --model <model>

# Test NumPy array utilities:
pixi run python src/python/array_utils.py --rows 5 --cols 10 --seed 42
```

### Testing
```bash
# Python tests:
pixi run pytest           # Run Python tests with pixi
pytest                    # Run Python tests with conda

# Test paths configured in pyproject.toml to use src/ and tests/
```

## Architecture

### Frontend (React + Tauri)
- **App.tsx** - Main application entry point with system info and matrix visualization
- **components/** - React components including D3Matrix for data visualization
- **src-tauri/** - Rust backend with Tauri commands for system information and GPU detection

### Python ML Pipeline
- **impute.py** - Main CLI entry point with argument parsing and model dispatching
- **bimamba/** - BiMamba state space model implementation for sequence imputation
- **modernBERT/** - ModernBERT transformer model for genetic sequence modeling
- **hmm/** - Hidden Markov Model implementation with Viterbi decoding
- **ps4g_io/** - Custom file format handling and PyTorch data loaders
- **bed_io/** - BED file format output for genomic data

### Key Data Flow
1. Input: PS4G format files (custom haplotype format)
2. Preprocessing: Convert to matrices with optional weighting schemes
3. Model Training/Inference: BiMamba, ModernBERT, or KNN models
4. Post-processing: Optional HMM smoothing with Viterbi decoding
5. Output: Extended BED format files

### Rust Backend Commands
- `greet` - Basic greeting command
- `gpu_adapters` - GPU detection and system information
- `greet_py` - Python integration example
- `get_sample_visualization_data` - Generates sample NumPy arrays for visualization testing
- `run_imputation_visualization` - Runs ML imputation and returns visualization data

### Python-TypeScript Data Transfer
- **Approach**: JSON + Base64 encoding for NumPy arrays
- **Python Utilities**: `src/python/array_utils.py` - Converts NumPy arrays to JSON format
- **TypeScript Utilities**: `src/utils/arrayUtils.ts` - Decodes base64 arrays and validates data
- **React Component**: `src/components/ArrayVisualization.tsx` - Interactive demo component
- **Supported Data Types**: float32, float64, int32, int16, uint8
- **Performance**: Efficient for arrays up to ~10MB, supports larger arrays via file-based transfer

## Model-Specific Notes

### BiMamba Model
- Uses state space models for sequence modeling
- Requires pre-trained model file: `src/bimamba_model.pth`
- Supports GPU acceleration when available
- Window-based processing with configurable window sizes

### ModernBERT Model
- Transformer-based approach for genetic imputation
- Requires pre-trained model file: `src/modernbert.pth`
- Encoder-only architecture optimized for imputation tasks

### File Format Handling
- **PS4G files** - Custom format for haplotype data with metadata
- **BED files** - Standard genomic interval format for output
- Support for weighted and unweighted data processing
- Optional collapsing of gamete sets and contiguous regions

## Development Environment

- **Python**: 3.10+ (managed via pixi or conda)
- **Node.js**: Required for frontend development
- **Rust**: Required for Tauri backend
- **GPU Support**: CUDA 12+ for Linux GPU acceleration
- **Key Dependencies**: PyTorch, transformers, mamba-ssm, D3.js, Tauri

## Testing Strategy

- Python tests located in `tests/python/`
- Focus on PS4G I/O and PyTorch data loaders
- Use pytest with coverage reporting
- PYTHONPATH configured to include `src/` directory

## Known Issues & Solutions

### Path Resolution in Tauri
- **Issue**: Tauri runs from `src-tauri/` directory, causing Python script path issues
- **Solution**: Auto-detect working directory and navigate to project root
- **Implementation**: Check if current directory is `src-tauri`, then use parent directory

### Python Environment Detection
- **Issue**: Need to use correct Python environment (pixi vs system)
- **Solution**: Auto-detect pixi availability and use appropriate command
- **Commands**: `pixi run python` when available, fallback to `python`