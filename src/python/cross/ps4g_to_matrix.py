import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import logging
from chrom_lengths import chrom_lengths
from python.ps4g_io.ps4g import load_ps4g_file, extract_metadata


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

def include_all_pos_inference(collapsed_matrix, unique_pos, length):
    '''
    Adds unlabelled bins to the collapsed matrix with -1 labels
    '''
    last_bin = length // 256

    all_pos_matrix = np.zeros((last_bin+1, collapsed_matrix.shape[1]-1))
    all_pos_labels = np.full((last_bin+1, 1), -1)
    all_pos_matrix = np.concatenate((all_pos_matrix, all_pos_labels), axis=1)
    all_pos_matrix[unique_pos, :] = collapsed_matrix

    return all_pos_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps4g-dir", type=str, help="directory containing PS4G matrices")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--collapse", type=bool, default=False, help="flag to collapse ps4g by position")
    parser.add_argument("--include-all-pos", type=bool, default=False, help="flag to include empty positions, must collapse")
    parser.add_argument("--ref-fasta", type=str, help="path to reference fasta (required for --include-all-pos to obtain chr lengths)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chr_lengths = chrom_lengths(args.ref_fasta)

    for ps4g_file in os.listdir(args.ps4g_dir):
        sample_name = ps4g_file.split("_ps4g")[0]
        assembly = ps4g_file.split("_")[0]

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
            matrix = create_chromosome_matrix_inference(ps4g_chr.reset_index(), gamete_to_idx)
            if args.collapse:
                collapsed_matrix, unique_pos = collapse_matrix_inference(matrix, ps4g_chr["refPosBinned"])
                if args.include_all_pos:
                    length = chr_lengths[chr]
                    all_pos_matrix = include_all_pos_inference(collapsed_matrix, unique_pos, length)
                    np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", all_pos_matrix)
                else:
                    np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", collapsed_matrix)
            else:
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", matrix)