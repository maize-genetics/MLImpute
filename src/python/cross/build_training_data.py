import pandas as pd
import numpy as np
from python.ps4g_io.ps4g import load_ps4g_file, extract_metadata
from tqdm import tqdm
import argparse
import os
import logging

def create_chromosome_matrix(ps4g, gamete_to_idx, answer_key):
    """
    Create a multihot encoded matrix [# entries, num_gametes + 1] from the PS4G data.
    0 = miss
    Read count = hit
    Last position represents the label
    Replace values >128 with 127 (max value for np.int8)

    Args:
        ps4g (pd.DataFrame): DataFrame containing the PS4G data for one chromosome.
        gamete_data (list): List of dictionaries containing gamete data.

    Returns:
        np.ndarray: A multihot encoded matrix.
    """

    # Get number of unique gametes
    num_classes = len(gamete_to_idx)

    X_multihot = np.zeros((len(ps4g), num_classes + 1), dtype=np.int8)

    for i, row in tqdm(enumerate(ps4g.itertuples()), total=len(ps4g)):
        count = min(row.count, 127)
        X_multihot[i, row.gameteSet] = count  # vectorized assignment
        if row.refPosBinned >= len(answer_key[row.refContig]): parent = None
        else: parent = answer_key[row.refContig][row.refPosBinned]

        X_multihot[i, -1] = gamete_to_idx.get(str(parent), -1) # convert parent to idx

    return X_multihot

def build_answer_key(keyfile):
    '''
    return a dictionary answer_key mapping chromosome to a list of labels, with indices corresponding to binned positions
    may include unlabelled bins
    '''
    key_df = (pd.read_csv(keyfile, sep="\t", header=None, usecols=[3, 4, 5, 6],
                         names=["ref_chr", "ref_start", "ref_end", "founder"]).drop_duplicates().sort_values(by=["ref_chr", "ref_start"]))

    # convert ref_chr, ref_start, ref_end (columns 1-3) to binned positions
    key_df["ref_start"] = key_df["ref_start"]//256
    key_df["ref_end"] = key_df["ref_end"]//256

    # convert to numpy array where index is binned position, fill in ranges with founder and empty bins with NA

    # Build per-chromosome arrays and fill
    answer_key = {}
    for chrom, grp in key_df.groupby("ref_chr", sort=False):
        max_bin = int(grp["ref_end"].max())
        # dtype=object so we can store founder strings; fill with NaN for empty
        arr = np.full(shape=max_bin, fill_value=np.nan, dtype=object)
        rows = grp.itertuples(index=False)
        for row in rows:
            # row has fields: ref_chr, ref_start, ref_end, founder, start_bp, end_bp, start_bin, end_bin
            s, e = int(row.ref_start), int(row.ref_end)
            if s < e:
                arr[s:e] = row.founder  # slice assignment is fast; "last" wins by default

        # If resolve == "first" we overwrote with earliest rows; nothing else needed
        answer_key[chrom] = arr

    return answer_key

def collapse_matrix(chrom_matrix, positions):
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
    collapsed_features = np.zeros((len(unique_pos), features.shape[1]), dtype=features.dtype)
    np.add.at(collapsed_features, inv_idx, features)

    collapsed_labels = labels[idx]

    # Combine features + labels back
    collapsed_matrix = np.column_stack([collapsed_features, collapsed_labels])

    return collapsed_matrix, unique_pos

def include_all_pos(collapsed_matrix, unique_pos, length):
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
    parser.add_argument("--assembly-key-dir", type=str, help="directory containing parent answer keys")
    parser.add_argument("--ps4g-dir", type=str, help="directory containing PS4G data")
    parser.add_argument("--output-dir", type=str, help="output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ps4g_file in os.listdir(args.ps4g_dir):
        sample_name = ps4g_file.split("_ps4g")[0]
        assembly = ps4g_file.split("_")[0]
        key_file = f"{assembly}_key.bed"

        try: ps4g_df = load_ps4g_file(f"{args.ps4g_dir}/{ps4g_file}")
        except Exception as e:
            logging.info(f"Error loading {ps4g_file}: {e}")
            raise

        logging.info("Loaded file")

        metadata, gamete_data = extract_metadata(f"{args.ps4g_dir}/{ps4g_file}")
        gamete_to_idx = {str(d["gamete"]): int(d["gamete_index"]) for d in gamete_data}
        logging.info("extracted metadata")

        key_dict = build_answer_key(f"{args.assembly_key_dir}/{key_file}")
        logging.info("created answer key")

        chromosomes = ps4g_df["refContig"].unique()

        for chr in chromosomes:
            ps4g_chr = ps4g_df[ps4g_df["refContig"] == chr]
            matrix = create_chromosome_matrix(ps4g_chr.reset_index(), gamete_to_idx, key_dict)
            # count number of -1

            # Separate data and labels
            X = matrix[:, :-1]
            y = matrix[:, -1].astype(int)

            # 1️⃣ Count of missing labels
            missing_mask = (y == -1)
            num_missing = missing_mask.sum()
            num_total = len(y)
            logging.info("percent unlabeled: ", num_missing/num_total)

            # 2️⃣ For rows with valid labels
            valid_mask = ~missing_mask
            valid_indices = np.where(valid_mask)[0]

            has_nonzero = []

            for i in valid_indices:
                label_idx = int(y[i])
                row = X[i, :]
                # does this row have a nonzero value at its label index?
                val = row[label_idx]
                has_nonzero.append(val > 0)

            has_nonzero = np.array(has_nonzero)
            logging.info("accuracy: ", has_nonzero.mean())

            np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", matrix.astype(np.int8))