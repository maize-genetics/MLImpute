import argparse
import logging
import time
import sys
from pathlib import Path
from ps4g_io.ps4g import convert_ps4g
from python.bimamba.bimamba_impute import run_bimamba_imputation
from python.modernBERT.modernBERT_impute import run_modernBERT_imputation
from bed_io.bed import output_predictions


# Example model imports (these would be your implementations)
# from models.knn import run_knn
# from models.mamba import run_mamba
# from models.bert import run_modernbert


def load_input(ps4g_file, weight="global", collapse=False):
    """
    Load the custom haplotype input file.
    Note we leave this in a numpy array as not every model uses torch.
    """
    logging.info(f"Loading input from {ps4g_file}")
    ps4g_data, weights = convert_ps4g(ps4g_file, weight, collapse)
    return ps4g_data, weights


def save_output(ps4g_file, output_path, results):
    """
    Save the imputed haplotypes to an extended BED format.
    """
    logging.info(f"Saving results to {output_path}")
    output_predictions(ps4g_file, output_path, results)

def run_model(args, data, weights):
    """
    Dispatch to the appropriate model based on the name.
    """
    model_name = args.model
    logging.info(f"Running model: {model_name}")

    if model_name == "knn":
        return {"rows": [["chr1", 100, 200, "A", "B"]]}  # replace with run_knn(data)
    elif model_name == "mamba":
        return run_bimamba_imputation(args,data, weights) # replace with run_mamba(data)
    elif model_name == "modernbert":
        return run_modernBERT_imputation(args, data, weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Haplotype Imputation Tool")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to output BED file")
    parser.add_argument("--model", "-m", choices=["knn", "mamba", "modernbert"], required=True, help="Imputation model")
    parser.add_argument("--weight", "-w", choices=["global", "unweighted"], default="global", help="Weighting strategy for PS4G data")
    parser.add_argument("--collapse", "-c", action="store_true", help="Collapse gamete sets into a single row per position")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--global-weights", type=str, default=None)
    parser.add_argument("--HMM", type=bool, default=False)
    parser.add_argument("--diploid", type=bool, default=False)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    try:
        start_time = time.time()

        # Load input data
        data, weights = load_input(args.input)

        # Run selected model
        results = run_model(args, data, weights)

        # Save output
        save_output(args.input, args.output, results)

        logging.info(f"Finished in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()