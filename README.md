# MLImpute
Simple tool to run Machine Learning based imputation techniques

## Prerequisites

* [pixi](https://pixi.sh/latest/installation/) (v0.5+)
* NVIDIA-based GPU hardware with CUDA (v12+)
  + For `gpu` environment only (_see next section_)
* Linux or macOS OS environment preferred


## Setup

**Retrieve project code**

```bash
git clone https://github.com/maize-genetics/MLImpute.git
cd MLImpute
```

**Option A (_preferred_) - set up virtual environment (pixi)**

```bash
# For CPU only
pixi install

# For Linux + GPU machines
pixi install --environment gpu
```

**Option B - set up virtual environment (conda)**

```bash
conda env create -f environment.yml
```

## Run (CLI tool)

Conda:
```bash
python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```

Pixi:
```bash
pixi run -- python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```

# Development (GUI app - _experimental_)

## Prerequisites

* Node.js
* Rust (with `wasm32-unknown-unknown` target for web builds)
* [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for web builds only)

## Tauri Desktop App

If running this for the first time:

```bash
# First install pixi
# Next, run the pixi install script
pixi install

# Initialize npm environment
npm install

# Set up wheelhouse
npm run download-python
npm run build-wheelhouse

# Run the dev container of the Tauri app
npm run tauri dev
```

For subsequent runs, just use `npm run tauri dev`.

## Static Web App (PS4G / BED Visualization)

The PS4G and BED visualization components can be built as a standalone
static web application. File parsing runs entirely client-side via Rust
compiled to WebAssembly -- no server required.

### One-time setup

```bash
# Add the WASM compilation target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install npm dependencies
npm install
```

### Development

```bash
# Start the Vite dev server in web mode (port 3001)
npm run dev:web
```

### Production build

```bash
# Build WASM + TypeScript + Vite to dist-web/
npm run build:web

# Preview the production build locally
npm run preview:web
```

The `dist-web/` directory is a fully static site that can be deployed to
GitHub Pages, Netlify, Vercel, S3, or any static file host.

### Build commands reference

| Command              | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `npm run dev`        | Vite dev server for Tauri (port 3000)                              |
| `npm run dev:web`    | Vite dev server for web (port 3001)                                |
| `npm run build`      | Production frontend build for Tauri (`dist/`)                      |
| `npm run build:web`  | Full web pipeline: WASM + TS + Vite (`dist-web/`)                  |
| `npm run build:wasm` | Build only the WASM module (`src/wasm/pkg/`)                       |
| `npm run preview:web`| Serve the `dist-web/` production build locally                     |
| `npm run tauri dev`  | Launch the full Tauri desktop app in development mode              |

# Testing

## Run python unit tests with HTML output (CLI)

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```





