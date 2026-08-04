import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import logging
from cross.chrom_lengths import chrom_lengths
from ps4g_io.ps4g import load_ps4g_file, extract_metadata


def create_chromosome_matrix_inference(ps4g, gamete_to_idx):
    """
    Create a multihot encoded matrix [# entries, num_gametes + 1] from the PS4G matrices.
    0 = miss
    Read count = hit
    Last position represents the label, for inference label is always -1 (unknown)

    Args:
        ps4g (pd.DataFrame): DataFrame containing the PS4G matrices for one chromosome.
        gamete_data (list): List of dictionaries containing gamete matrices.

    Returns:
        np.ndarray: A multihot encoded matrix.
    """

    # Get number of unique gametes
    num_classes = len(gamete_to_idx)

    X_multihot = np.zeros((len(ps4g), num_classes + 1), dtype=np.int8)

    for i, row in tqdm(enumerate(ps4g.itertuples()), total=len(ps4g)):
        count = min(row.count, 127) # clip to max int8 value
        X_multihot[i, row.gameteSet] = count  # vectorized assignment
        X_multihot[i, -1] = -1 # all values are unlabelled at inference

    return X_multihot

def collapse_matrix_inference(chrom_matrix, positions):
    """
    Collapse rows in a NumPy matrix by summing rows with the same binned position,
    excluding the last column (labels).

    Args:
        chrom_matrix (np.ndarray): shape (n_rows, n_cols)
            Feature matrix where the last column represents labels.
        positions (list or array-like): length n_rows
            Binned positions corresponding to each row.

    Returns:
        collapsed_matrix (np.ndarray): shape (n_unique_bins, n_cols)
            Collapsed features with the same labels appended.
        collapsed_positions (np.ndarray): shape (n_unique_bins,)
    """
    positions = np.asarray(positions)

    # Separate features and labels
    features = chrom_matrix[:, :-1]
    labels = chrom_matrix[:, -1]

    # Find unique positions and mapping
    unique_pos, idx, inv_idx = np.unique(positions, return_index=True, return_inverse=True)

    # Collapse features by summing
    # increase int size to account for possible int8 overflow
    collapsed_features = np.zeros((len(unique_pos), features.shape[1]), dtype=np.int32)
    np.add.at(collapsed_features, inv_idx, features.astype(np.int32))
    # revert back to int8
    collapsed_features = np.clip(collapsed_features, 0, 127).astype(np.int8)
    collapsed_labels = labels[idx]

    # Combine features + labels back
    collapsed_matrix = np.column_stack([collapsed_features, collapsed_labels])

    return collapsed_matrix, unique_pos

def include_all_pos_inference(collapsed_matrix, unique_pos, length, bin_size=256):
    '''
    Adds unlabelled bins to the collapsed matrix with -1 labels
    '''
    last_bin = length // bin_size

    all_pos_matrix = np.zeros((last_bin+1, collapsed_matrix.shape[1]-1))
    all_pos_labels = np.full((last_bin+1, 1), -1)
    all_pos_matrix = np.concatenate((all_pos_matrix, all_pos_labels), axis=1)
    all_pos_matrix[unique_pos, :] = collapsed_matrix
    all_positions = np.arange(last_bin+1)
    return all_pos_matrix, all_positions

def normalize_positions(positions, method='minmax'):
    """Normalize genomic positions for positional embedding"""
    positions = np.array(positions)

    if method == 'minmax':
        # Scale to [0, 1]
        pos_min, pos_max = positions.min()
        pos_max = positions.max()
        if pos_max > pos_min:
            return (positions - pos_min) / (pos_max - pos_min)
        else:
            return np.zeros_like(positions)
    elif method == 'zscore':
        # Standardize to mean=0, std=1
        return (positions - positions.mean()) / (positions.std() + 1e-8)
    elif method == 'log':
        # Log scale (useful for genomic distances)
        return np.log10(positions + 1) / np.log10(positions.max() + 1)

def add_positional_embedding(matrix, positions, method='minmax', diploid=False):
    """Add positional information as an additional column"""
    pos_normalized = normalize_positions(positions, method=method)
    pos_int8 = (pos_normalized * 127).astype(np.int8)

    # Reshape to column vector if needed
    if pos_int8.ndim == 1:
        pos_int8 = pos_int8.reshape(-1, 1)

    # Separate features and labels
    if diploid:  # diploid: features + 2 labels
        features = matrix[:, :-2]
        labels = matrix[:, -2:]
    else:  # haploid: features + 1 label
        features = matrix[:, :-1]
        labels = matrix[:, -1:]

    # Insert position AFTER features, BEFORE labels
    return np.concatenate([features, pos_int8, labels], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps4g-dir", type=str, help="directory containing PS4G matrices")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--collapse", action='store_true', help="flag to collapse ps4g by position, only flag --collapse or --include-all-pos")
    parser.add_argument("--include-all-pos", action='store_true', help="flag to include empty positions, will also be collapsed")
    parser.add_argument("--positional-embed", action='store_true', help="flag to include positional embed")
    parser.add_argument("--ref-fasta", type=str, help="path to reference fasta (required for --include-all-pos to obtain chr lengths)")
    parser.add_argument("--bin-size", type=int, default=256, help="bin size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chr_lengths = chrom_lengths(args.ref_fasta)

    for ps4g_file in os.listdir(args.ps4g_dir):
        sample_name = ps4g_file.replace(".ps4g", "").replace("_ps4g.txt", "")

        try: ps4g_df = load_ps4g_file(f"{args.ps4g_dir}/{ps4g_file}")
        except Exception as e:
            logging.info(f"Error loading {ps4g_file}: {e}")
            raise

        logging.info("Loaded file")

        metadata, gamete_data = extract_metadata(f"{args.ps4g_dir}/{ps4g_file}")
        gamete_to_idx = {str(d["gamete"]): int(d["gamete_index"]) for d in gamete_data}
        logging.info("extracted metadata")

        chromosomes = ps4g_df["refContig"].unique()

        for chr in chromosomes:
            ps4g_chr = ps4g_df[ps4g_df["refContig"] == chr]
            positions = ps4g_chr["refPosBinned"]
            matrix = create_chromosome_matrix_inference(ps4g_chr.reset_index(), gamete_to_idx)

            if args.collapse:
                collapsed_matrix, unique_pos = collapse_matrix_inference(matrix, positions)
                if args.positional_embed:
                    collapsed_matrix = add_positional_embedding(collapsed_matrix, unique_pos)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", collapsed_matrix)
            elif args.include_all_pos:
                collapsed_matrix, unique_pos = collapse_matrix_inference(matrix, positions)
                length = chr_lengths[chr]
                all_pos_matrix, all_positions = include_all_pos_inference(collapsed_matrix, unique_pos, length, bin_size=args.bin_size)
                if args.positional_embed:
                    all_pos_matrix = add_positional_embedding(all_pos_matrix, all_positions)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", all_pos_matrix)
            else:
                if args.positional_embed:
                    matrix = add_positional_embedding(matrix, positions)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", matrix)