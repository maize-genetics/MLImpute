import argparse
import logging
import time
import sys
from pathlib import Path
from python.ps4g_io.ps4g import convert_ps4g
#from python.bimamba.bimamba_impute import run_bimamba_imputation
from python.modernBERT.modernBERT_impute import run_modernBERT_imputation
from python.bed_io.bed import output_predictions
from python.knn.knn import run_knn
from python.array_utils import create_visualization_response, create_error_response
import numpy as np


def load_input(ps4g_file, weight="global", collapse=False):
    """
    Load the custom haplotype input file.
    Note we leave this in a numpy array as not every model uses torch.
    """
    logging.info(f"Loading input from {ps4g_file}")
    ps4g_data, weights = convert_ps4g(str(ps4g_file), weight, collapse)
    return ps4g_data, weights


def save_output(ps4g_file, output_path, results, collapse_bed_regions=True):
    """
    Save the imputed haplotypes to an extended BED format.
    """
    logging.info(f"Saving results to {output_path}")
    output_predictions(ps4g_file, output_path, results, collapse_bed_regions)

def run_model(args, data, weights):
    """
    Dispatch to the appropriate model based on the name.
    """
    model_name = args.model
    logging.info(f"Running model: {model_name}")

    if model_name == "knn":
        return run_knn(data, args.window_size, args.diploid)
    elif model_name == "mamba":
        return run_bimamba_imputation(args, data, weights)
    elif model_name == "modernbert":
        return run_modernBERT_imputation(args, data, weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_visualization_data(ps4g_data, results, weights=None):
    """
    Create visualization data from the imputation results.
    
    Args:
        ps4g_data: Original input data matrix
        results: Imputation results 
        weights: Optional weights array
        
    Returns:
        JSON string with visualization data
    """
    try:
        # Create a comparison matrix showing original vs imputed data
        # For visualization, we'll show the imputation results as a heatmap
        
        # Convert results to a matrix format suitable for visualization
        if len(results.shape) == 2:
            viz_matrix = results.astype(np.float32)
        else:
            # If results are indices, create a categorical visualization
            viz_matrix = results.astype(np.float32)
        
        # Create row and column labels
        num_positions = viz_matrix.shape[0]
        num_samples = viz_matrix.shape[1] if len(viz_matrix.shape) > 1 else 1
        
        row_labels = [f"Pos_{i}" for i in range(num_positions)]
        col_labels = [f"Parent_{i}" for i in range(num_samples)] if num_samples > 1 else ["Result"]
        
        # Add metadata about the imputation
        metadata = {
            "type": "imputation_results",
            "original_shape": list(ps4g_data.shape),
            "result_shape": list(viz_matrix.shape),
            "has_weights": weights is not None,
            "description": "Imputation results showing predicted parent assignments"
        }
        
        return create_visualization_response(viz_matrix, row_labels, col_labels, metadata)
        
    except Exception as e:
        return create_error_response(f"Failed to create visualization data: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Haplotype Imputation Tool with Visualization")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to output BED file")
    parser.add_argument("--model", "-m", choices=["knn", "mamba", "modernbert"], required=True, help="Imputation model")
    parser.add_argument("--weight", "-w", choices=["global", "unweighted"], default="global", help="Weighting strategy for PS4G data")
    parser.add_argument("--collapse", "-c", action="store_true", help="Collapse gamete sets into a single row per position")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--global-weights", type=str, default=None)
    parser.add_argument("--HMM", type=bool, default=False)
    parser.add_argument("--diploid", type=bool, default=False)
    parser.add_argument("--window-size", type=int, default=21, help="Size of the sliding window for KNN model (must be odd)")

    parser.add_argument("--collapse-bed", action="store_true", help="Collapse contiguous BED regions in output")
    parser.add_argument("--viz-only", action="store_true", help="Only output visualization data (JSON) to stdout")
    
    args = parser.parse_args()

    # Configure logging to stderr so it doesn't interfere with JSON output
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr
    )

    try:
        start_time = time.time()

        # Load input data
        data, weights = load_input(args.input, args.weight, args.collapse)

        # Run selected model
        results = run_model(args, data, weights)

        if args.viz_only:
            # Output only visualization data as JSON
            viz_data = create_visualization_data(data, results, weights)
            print(viz_data)
        else:
            # Save BED output and also output visualization data
            save_output(args.input, args.output, results, args.collapse_bed)
            
            # Create and output visualization data
            viz_data = create_visualization_data(data, results, weights)
            
            # Output both success message and visualization data
            execution_time = time.time() - start_time
            success_msg = {
                "status": "success",
                "message": f"Imputation completed in {execution_time:.2f} seconds",
                "bed_file": str(args.output),
                "visualization_data": viz_data
            }
            
            import json
            print(json.dumps(success_msg))

        logging.info(f"Finished in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        error_msg = create_error_response(f"Imputation failed: {str(e)}")
        if args.viz_only:
            print(error_msg)
        else:
            print(error_msg)
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()