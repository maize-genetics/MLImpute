# MLImpute
Simple tool to run Machine Learning based imputation techniques


# Installation Conda Environment

```bash
conda env create -f environment.yml
```

# Installation using pixi

Install pixi if you haven't already:

```bash
curl -sSf https://pixi.sh/install.sh | bash
```

Then, you can install the MLImpute package using:

```bash
pixi install
```


# Run CLI Script

Conda:
```bash
python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```

Pixi:
```bash
pixi run -- python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```