# GRITS
**G**enetic **R**ecombination **I**mputation **T**ool **S**et

## Prerequisites

* [pixi](https://pixi.sh/latest/installation/) (v0.5+)
* NVIDIA-based GPU hardware with CUDA (v12+)
  + For `gpu` environment only (_see next section_)
* Linux or macOS OS environment preferred


## Setup

**Retrieve project code**

```bash
git clone https://github.com/maize-genetics/grits.git
cd grits
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
compiled to WebAssembly - no server required.

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

## Hosting the Web App with GRITS Documentation

The deployed site (for example, GitHub Pages) serves the WASM web app at the
root and the [MkDocs](https://www.mkdocs.org/) documentation under `/docs/`.
The docs are built from this project's markdown in `docs/` plus pages pulled
from the accompanying [`seq_sim`](https://github.com/maize-genetics/seq_sim)
repository at build time. You can reproduce this combined site locally two
ways.

### One-time setup

In addition to the [Static Web App one-time setup](#one-time-setup) above,
install the `docs` pixi environment (used to build the documentation):

```bash
pixi install --environment docs
```

> **Note:** Building the docs fetches a few files from the `seq_sim` repository
> over the network, so an internet connection is required. Override the source
> ref with `SEQ_SIM_REF=<branch|tag|sha>` if needed.

### Option A - Live dev servers (fast iteration)

Run the web app and the docs as two live-reloading servers. In web mode, the
Vite dev server proxies `/docs` to the local MkDocs server, so the in-app
"Documentation" link works end to end.

```bash
# Terminal 1 - serve the docs at http://127.0.0.1:8000
pixi run -e docs docs-serve

# Terminal 2 - serve the web app at http://localhost:3001
npm run dev:web
```

Open http://localhost:3001 and click **Documentation** (or browse to
http://localhost:3001/docs/). Both servers hot-reload on changes.

### Option B - Combined production preview (matches deployment)

Build both pieces, combine them into a single static tree, and preview it. This
mirrors exactly what the GitHub Pages workflow (`.github/workflows/pages.yml`)
deploys, so the docs are served from the built files rather than a proxy.

```bash
# 1. Build the WASM web app into dist-web/
npm run build:web

# 2. Build the documentation site into site/ (fetches seq_sim docs first)
pixi run -e docs docs-build

# 3. Combine the docs into the web app output under dist-web/docs/
mkdir -p dist-web/docs && cp -r site/. dist-web/docs/

# 4. Preview the combined static site at http://localhost:4173
npm run preview:web
```

Open http://localhost:4173 for the web app and http://localhost:4173/docs/ for
the documentation. No MkDocs server needs to be running for the preview; the
docs are served as static files from `dist-web/docs/`.

### Build commands reference

| Command                       | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| `npm run dev`                 | Vite dev server for Tauri (port 3000)                                  |
| `npm run dev:web`             | Vite dev server for web (port 3001); proxies `/docs` to MkDocs on 8000 |
| `npm run build`               | Production frontend build for Tauri (`dist/`)                          |
| `npm run build:web`           | Full web pipeline: WASM + TS + Vite (`dist-web/`)                      |
| `npm run build:web:fast`      | Web build skipping the WASM step (reuses `src/wasm/pkg/`)              |
| `npm run build:wasm`          | Build only the WASM module (`src/wasm/pkg/`)                           |
| `npm run preview:web`         | Serve the `dist-web/` production build locally (port 4173)             |
| `npm run tauri dev`           | Launch the full Tauri desktop app in development mode                  |
| `pixi run -e docs docs-serve` | Live-reload MkDocs docs at http://127.0.0.1:8000                       |
| `pixi run -e docs docs-build` | Build the docs into `site/` (fetches `seq_sim` docs first)             |



# Testing

## Run python unit tests with HTML output (CLI)

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```





