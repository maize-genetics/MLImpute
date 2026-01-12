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
* Rust

## Run Tauri dev container

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

For subsequent runs, just use `npm run tauri dev`

# Testing

## Run python unit tests with HTML output (CLI)

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```





