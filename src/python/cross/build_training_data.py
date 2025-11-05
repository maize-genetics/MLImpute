import pandas as pd
import numpy as np
import logging
from MLImpute.src.python.ps4g_io.ps4g import load_ps4g_file, extract_metadata
from tqdm import tqdm


def create_chromosome_matrix(ps4g, gamete_to_idx, answer_key):
    """
    Create a multihot encoded matrix [# entries, num_gametes + 1] from the PS4G data.
    0 = miss
    Read count = hit
    Last position represents the label

    Args:
        ps4g (pd.DataFrame): DataFrame containing the PS4G data for one chromosome.
        gamete_data (list): List of dictionaries containing gamete data.

    Returns:
        np.ndarray: A multihot encoded matrix.
    """

    # Get number of unique gametes
    num_classes = len(gamete_to_idx)

    X_multihot = np.zeros((len(ps4g), num_classes + 1), dtype=np.float32)

    for i, row in tqdm(enumerate(ps4g.itertuples()), total=len(ps4g)):
        X_multihot[i, row.gameteSet] = row.count  # vectorized assignment
        if row.refPosBinned >= len(answer_key[row.refContig]): parent = None
        else: parent = answer_key[row.refContig][row.refPosBinned] # TODO: what if idx is out of bounds?

        X_multihot[i, -1] = gamete_to_idx.get(str(parent), -1) # convert parent to idx

    # need to assign labels

    return X_multihot

def build_answer_key(keyfile):
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


if __name__ == '__main__':
    ps4g_file = "B97_1_ps4g.txt"
    key_file = "B97_key.bed"

    sample_name = ps4g_file.split("_ps4g")[0]

    ps4g_df = load_ps4g_file(ps4g_file)
    print("loaded file")

    metadata, gamete_data = extract_metadata(ps4g_file)
    gamete_to_idx = {str(d["gamete"]): int(d["gamete_index"]) for d in gamete_data}
    print("extracted metadata")

    key_dict = build_answer_key(key_file)
    print("created answer key")

    chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10"]

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
        print("percent unlabeled: ", num_missing/num_total)

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
        print("accuracy: ", has_nonzero.mean())

        #np.save(f"{sample_name}_{chr}_matrix.npy", matrix.astype(np.int8))
