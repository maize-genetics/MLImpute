import pandas as pd
import numpy as np
from src.python.ps4g_io.ps4g import load_ps4g_file, extract_metadata
from tqdm import tqdm
import argparse
import os
import logging

def create_chromosome_matrix(ps4g, gamete_to_idx, answer_key1, answer_key2=None):
    """
    Create a multihot encoded matrix [# entries, num_gametes + 1 (or 2)] from the PS4G data.
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
    if answer_key2 is None:
        X_multihot = np.zeros((len(ps4g), num_classes + 1), dtype=np.int8)
    else:
        X_multihot = np.zeros((len(ps4g), num_classes + 2), dtype=np.int8)

    for i, row in tqdm(enumerate(ps4g.itertuples()), total=len(ps4g)):
        count = min(row.count, 127)
        X_multihot[i, row.gameteSet] = count  # vectorized assignment

        if row.refPosBinned >= len(answer_key1[row.refContig]): parent1 = None
        else: parent1 = answer_key1[row.refContig][row.refPosBinned]

        if answer_key2 is not None:
            if row.refPosBinned >= len(answer_key2[row.refContig]): parent2 = None
            else: parent2 = answer_key2[row.refContig][row.refPosBinned]
            X_multihot[i, -2] = gamete_to_idx.get(str(parent2), -1)  # convert parent to idx

        X_multihot[i, -1] = gamete_to_idx.get(str(parent1), -1) # convert parent to idx

    return X_multihot

def build_answer_key(keyfile):
    '''
    return a dictionary answer_key mapping chromosome to a list of labels, with indices corresponding to binned positions
    may include unlabelled bins
    '''
    # TODO: v1 usecols=[3, 4, 5, 6] because using _key.bed, v2 use 1, 2, 3, 4 because _refkey.bed
    key_df = (pd.read_csv(keyfile, sep="\t", header=None,
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

def collapse_matrix(chrom_matrix, positions, diploid=False):
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
    if diploid:
        features = chrom_matrix[:, :-2]
        labels = chrom_matrix[:, -2:]
    else:
        features = chrom_matrix[:, :-1]
        labels = chrom_matrix[:, -1]

    # Find unique positions and mapping
    unique_pos, idx, inv_idx = np.unique(positions, return_index=True, return_inverse=True)

    # Collapse features by summing
    # increase int size to account for possible int8 overflow
    collapsed_features = np.zeros((len(unique_pos), features.shape[1]), dtype=np.int32)
    np.add.at(collapsed_features, inv_idx, features.astype(np.int32))
    # revert back to int8
    collapsed_features = np.clip(collapsed_features, -128, 127).astype(np.int8)
    collapsed_labels = labels[idx]

    # Combine features + labels back
    collapsed_matrix = np.column_stack([collapsed_features, collapsed_labels])

    return collapsed_matrix, unique_pos

def include_all_pos(collapsed_matrix, unique_pos, gamete_to_idx, answer_key1, answer_key2=None):
    '''
    Adds unlabelled bins to the collapsed matrix with -1 labels
    '''
    if answer_key2 is None:
        last_bin = len(answer_key1)
    else:
        last_bin = min(len(answer_key1), len(answer_key2))

    all_positions = np.arange(last_bin)
    all_pos_matrix = np.zeros((last_bin, collapsed_matrix.shape[1]), dtype=np.int8)
    labels_1 = np.array([gamete_to_idx.get(str(x), -1) for x in answer_key1], dtype=np.int8)
    all_pos_labels_1 = labels_1[:last_bin]
    if answer_key2 is not None:
        labels_2 = np.array([gamete_to_idx.get(str(x), -1) for x in answer_key2], dtype=np.int8)
        all_pos_labels_2 = labels_2[:last_bin]
        all_pos_matrix[:, -2] = all_pos_labels_2
    all_pos_matrix[:, -1] = all_pos_labels_1

    all_pos_matrix[unique_pos, ...] = collapsed_matrix

    return all_pos_matrix, all_positions

def normalize_positions(positions, method='minmax'):
    """Normalize genomic positions for positional embedding"""
    positions = np.array(positions)

    if method == 'minmax':
        # Scale to [0, 1]
        pos_min, pos_max = positions.min(), positions.max()
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
    parser.add_argument("--assembly-key-dir", type=str, help="directory containing parent answer keys")
    parser.add_argument("--ps4g-dir", type=str, help="directory containing PS4G data")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--collapse", action='store_true', help="flag to collapse ps4g by position, only flag --collapse or --include-all-pos")
    parser.add_argument("--include-all-pos", action='store_true', help="flag to include empty positions, will also be collapsed")
    parser.add_argument("--positional-embed", action='store_true', help="flag to include positional embed")
    parser.add_argument("--sample-assembly-key", type=str, help="""\
                                                                                tab separated file
                                                                                [sample] [assembly1] (optional) [assembly2]
                                                                                {sample}_ps4g.txt
                                                                                {assembly1}_key.bed
                                                                                {assembly2}_key.bed
                                                                                """)
    args = parser.parse_args()
    if args.collapse and args.include_all_pos:
        raise ValueError("Cannot use both --collapse and --include-all-pos flags simultaneously")

    os.makedirs(args.output_dir, exist_ok=True)

    sample_asm_key = pd.read_csv(args.sample_assembly_key, sep="\t", header=None)

    for ps4g_file in os.listdir(args.ps4g_dir):
        # TODO: old version is _ps4g.txt
        sample_name = ps4g_file.split(".ps4g")[0]
        row = sample_asm_key[sample_asm_key[0] == sample_name]
        if row.empty:
            raise ValueError(f"Sample {sample_name} not found in sample-assembly-key file.")
        assembly1 = row[1].iloc[0]
        # TODO: v1 should be _key.bed, v2 is _refkey.bed
        key_file1 = f"{assembly1}_refkey.bed"
        if sample_asm_key.shape[1] > 2:
            assembly2 = row[2].iloc[0]
            key_file2 = f"{assembly2}_refkey.bed"
            diploid = True
        else:
            key_file2 = None
            diploid = False

        try: ps4g_df = load_ps4g_file(f"{args.ps4g_dir}/{ps4g_file}")
        except Exception as e:
            logging.info(f"Error loading {ps4g_file}: {e}")
            raise

        logging.info("Loaded file")

        metadata, gamete_data = extract_metadata(f"{args.ps4g_dir}/{ps4g_file}")
        gamete_to_idx = {str(d["gamete"]): int(d["gamete_index"]) for d in gamete_data}
        logging.info("extracted metadata")

        key_dict1 = build_answer_key(f"{args.assembly_key_dir}/{key_file1}")
        if diploid:
            key_dict2 = build_answer_key(f"{args.assembly_key_dir}/{key_file2}")
        else:
            key_dict2 = None

        logging.info("created answer key")

        chromosomes = ps4g_df["refContig"].unique()

        for chr in chromosomes:
            ps4g_chr = ps4g_df[ps4g_df["refContig"] == chr]
            matrix = create_chromosome_matrix(ps4g_chr.reset_index(), gamete_to_idx, key_dict1, key_dict2)
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
            positions = ps4g_chr["refPosBinned"]

            if args.collapse:
                collapsed_matrix, unique_pos = collapse_matrix(matrix, positions, diploid)
                if args.positional_embed:
                    collapsed_matrix = add_positional_embedding(collapsed_matrix, unique_pos, diploid=diploid)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", collapsed_matrix)
            elif args.include_all_pos:
                collapsed_matrix, unique_pos = collapse_matrix(matrix, positions, diploid)
                if diploid:
                    all_pos_matrix, all_positions = include_all_pos(collapsed_matrix, unique_pos, gamete_to_idx, key_dict1[chr], key_dict2[chr])
                else:
                    all_pos_matrix, all_positions = include_all_pos(collapsed_matrix, unique_pos, gamete_to_idx, key_dict1[chr], answer_key2=None)
                if args.positional_embed:
                    all_pos_matrix = add_positional_embedding(all_pos_matrix, all_positions, diploid=diploid)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", all_pos_matrix)
            else:
                if args.positional_embed:
                    matrix = add_positional_embedding(matrix, positions, diploid=diploid)
                np.save(f"{args.output_dir}/{sample_name}_{chr}_matrix.npy", matrix)