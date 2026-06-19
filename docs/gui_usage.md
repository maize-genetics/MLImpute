# GUI Usage Guide

GRITS ships a cross-platform desktop application built with [Tauri](https://tauri.app/)
(Rust backend + React/TypeScript frontend). It provides an interactive interface
for running imputation models and exploring PS4G/BED data with D3-powered
visualizations, backed by an embedded Python runtime.

## Installing a prebuilt release

Download the latest installer from the
[GRITS releases page](https://github.com/maize-genetics/grits/releases):

- **macOS** - `.dmg`
- **Windows** - `.msi`

!!! note
    A prebuilt Linux GUI release is not currently provided. Linux users can build
    the application locally with the instructions below.

After installation the app bundles its own Python runtime and wheelhouse, so no
separate Python or pixi installation is required to run a released build.

## Building locally (development)

Requirements:

- [Node.js](https://nodejs.org/) (LTS)
- [Rust](https://www.rust-lang.org/tools/install) toolchain
- [pixi](https://pixi.sh/latest/) for the Python environment

```bash
git clone https://github.com/maize-genetics/grits.git
cd grits

# Install frontend and Python dependencies
npm install
pixi install

# Launch the app in development mode (hot reload)
npm run tauri dev
```

## Building for production

```bash
# Build the frontend, then bundle the desktop app
npm run build && npm run tauri build
```

Platform-specific installers are written to `src-tauri/target/release/bundle/`.
See the [Packaging](notes/packaging.md) guide for details on the self-contained
Python runtime and wheelhouse bundling.

## Application features

- **Interactive matrix visualization** - real-time D3 heatmaps of PS4G/BED data
  with zoom and brushing.
- **Imputation interface** - select a model (KNN, BiMamba, ModernBERT) and
  configure options such as weighting and gamete collapsing.
- **System information** - GPU detection and capability reporting via `wgpu`.
- **Native file dialogs** - open input files and choose output locations through
  the OS file picker.
- **Resizable sidebar** - adjustable workspace layout for controls and settings.

## How the GUI runs imputation

The Rust backend exposes a `run_python_imputation` command that invokes
`src/python/impute_with_viz.py`. It auto-detects whether to use `pixi run python`
or system `python`, resolves the project root (handling the `src-tauri/` working
directory), and streams visualization data back to the frontend.

The Python side encodes NumPy arrays as base64 JSON (`src/python/array_utils.py`),
which the frontend decodes into typed arrays (`src/utils/arrayUtils.ts`) for the
D3 matrix components.

## Troubleshooting

- **App cannot find Python**: ensure `pixi install` has been run in the project
  root for development builds. Released builds use the embedded runtime and do
  not require this.
- **macOS "app is damaged" warning**: releases use ad-hoc code signing; you may
  need to allow the app in System Settings -> Privacy & Security.
- **GPU not detected**: GPU acceleration (BiMamba, ModernBERT) requires CUDA 12+
  on Linux/Windows; macOS runs models on CPU.
