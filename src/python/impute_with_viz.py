#!/usr/bin/env python3
"""
Wrapper script that runs imputation and generates visualization data.

This script combines the main imputation functionality with visualization data generation
for the parent path visualization system.
"""

import argparse
import json
import logging
import sys
import tempfile
import numpy as np
from pathlib import Path

# Import the main imputation functionality
from impute import main as run_imputation, load_input
from array_utils import create_visualization_response, create_error_response

def parse_bed_file(bed_file_path):
    """
    Parse BED file to extract parent paths information.
    
    Args:
        bed_file_path: Path to the BED file output from imputation
        
    Returns:
        Dictionary with parent path information suitable for visualization
    """
    try:
        with open(bed_file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        if not data_lines:
            return None
        
        # Check if first line is a header by looking for expected column names
        if data_lines[0].startswith('chrom_idx') or 'parent1' in data_lines[0]:
            data_lines = data_lines[1:]  # Skip header
            
        positions = []
        parent1_samples = []
        parent2_samples = []
        
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) >= 4:  # chrom_idx, pos, parent1, parent2
                # Create position identifier from chrom_idx and pos
                chrom_idx = parts[0]
                pos = parts[1]
                pos_name = f"ChrIdx{chrom_idx}_{pos}"
                positions.append(pos_name)
                
                # Extract parent information (columns 2 and 3 are parent1 and parent2)
                parent1 = parts[2] if parts[2] else f"Unknown_{len(positions)}"
                parent2 = parts[3] if parts[3] else f"Unknown_{len(positions)}"
                
                parent1_samples.append(parent1)
                parent2_samples.append(parent2)
        
        # Get unique samples
        all_samples = list(set(parent1_samples + parent2_samples))
        all_samples.sort()  # Consistent ordering
        
        # Create matrix where rows are samples and columns are positions
        matrix = np.random.rand(len(all_samples), len(positions)).astype(np.float32)
        
        # Create parent path data
        parent1_path = []
        parent2_path = []
        highlights = []
        
        for i, (pos, p1_sample, p2_sample) in enumerate(zip(positions, parent1_samples, parent2_samples)):
            # Find row indices for the samples
            p1_row_idx = all_samples.index(p1_sample) if p1_sample in all_samples else 0
            p2_row_idx = all_samples.index(p2_sample) if p2_sample in all_samples else 0
            
            parent1_path.append({
                'position': pos,
                'sample': p1_sample,
                'row_idx': p1_row_idx,
                'col_idx': i
            })
            
            parent2_path.append({
                'position': pos,
                'sample': p2_sample,
                'row_idx': p2_row_idx,
                'col_idx': i
            })
            
            highlights.append({
                'row': p1_sample,
                'col': pos,
                'parent': 'parent1'
            })
            
            highlights.append({
                'row': p2_sample,
                'col': pos,
                'parent': 'parent2'
            })
            
            # Enhance matrix values at parent locations
            matrix[p1_row_idx, i] = 0.8 + 0.2 * np.random.random()
            matrix[p2_row_idx, i] = 0.1 + 0.2 * np.random.random()
        
        return {
            'matrix': matrix,
            'row_labels': all_samples,
            'col_labels': positions,
            'parent1_path': parent1_path,
            'parent2_path': parent2_path,
            'metadata': {
                'type': 'parent_paths',
                'description': f'Imputation results: parent paths through {len(all_samples)} samples across {len(positions)} positions',
                'source': 'imputation',
                'highlights': highlights
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to parse BED file {bed_file_path}: {e}")
        return None


def run_imputation_with_visualization(args):
    """
    Run imputation and generate visualization data.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        JSON string containing imputation results and visualization data
    """
    try:
        # First, run the main imputation process
        logging.info("Starting imputation process...")
        
        # Temporarily redirect stdout to capture any output from main imputation
        original_argv = sys.argv
        sys.argv = [
            'impute_with_viz.py',
            '--input', str(args.input),
            '--output', str(args.output), 
            '--model', args.model
        ]
        
        # Add optional arguments
        if args.weight:
            sys.argv.extend(['--weight', args.weight])
        if args.collapse:
            sys.argv.append('--collapse')
        if args.verbose:
            sys.argv.append('--verbose')
        if args.global_weights:
            sys.argv.extend(['--global-weights', args.global_weights])
        if args.HMM:
            sys.argv.extend(['--HMM', str(args.HMM)])
        if args.diploid:
            sys.argv.extend(['--diploid', str(args.diploid)])
        if args.collapse_bed:
            sys.argv.append('--collapse-bed')
        
        imputation_success = True
        try:
            # Run the main imputation
            run_imputation()
            logging.info("Imputation completed successfully")
            
        except SystemExit as e:
            if e.code != 0:
                logging.warning(f"Imputation failed with exit code {e.code}, will try to use existing output file if available")
                imputation_success = False
            
        finally:
            # Restore original argv
            sys.argv = original_argv
        
        # Check if output file exists (either newly created or pre-existing)
        if not args.output.exists():
            # If no BED file was created, return an error
            logging.error("No BED output file found - imputation may have failed")
            return json.dumps({
                'success': False,
                'message': 'Imputation failed - no output file was created',
                'output_file': None,
                'visualization_data': create_error_response('No BED output file was created')
            })
        else:
            # Parse the BED file to extract parent path information
            logging.info(f"Parsing BED output file: {args.output}")
            bed_data = parse_bed_file(args.output)
            
            if bed_data:
                visualization_data = create_visualization_response(
                    bed_data['matrix'],
                    bed_data['row_labels'],
                    bed_data['col_labels'], 
                    bed_data['metadata']
                )
            else:
                # Return an error if BED parsing fails
                logging.error("Failed to parse BED file")
                return json.dumps({
                    'success': False,
                    'message': 'Imputation completed but failed to parse output BED file',
                    'output_file': str(args.output) if args.output.exists() else None,
                    'visualization_data': create_error_response('Failed to parse BED file for visualization')
                })
        
        # Return the results in the expected format
        result = {
            'success': imputation_success,
            'message': 'Imputation and visualization completed successfully' if imputation_success else 'Imputation had issues but output file was processed for visualization',
            'output_file': str(args.output) if args.output.exists() else None,
            'visualization_data': visualization_data
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Error during imputation with visualization: {e}")
        return json.dumps({
            'success': False,
            'message': f'Error: {e}',
            'output_file': None,
            'visualization_data': create_error_response(str(e))
        })


def main():
    parser = argparse.ArgumentParser(description="Haplotype Imputation with Visualization")
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
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    # Run imputation with visualization
    result = run_imputation_with_visualization(args)
    print(result)


if __name__ == "__main__":
    main()