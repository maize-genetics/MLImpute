import numpy as np
import pandas as pd
import torch
import math
import argparse
import torch.nn.functional as F
from python.hmm.viterbi import build_pair_states, viterbi_decode
from python.ps4g_io.ps4g import build_index_lookup


def run_hmm_imputation(args):
    # check to see if the required arguments are provided
    if args.input_path is None or args.output_bed is None or args.global_weights is None or args.ps4g_file is None:
        raise ValueError("Missing required arguments: --input-path, --output-bed, --global-weights, --ps4g-file")

    window_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_paths = [args.input_path]

    # Reconstruct test_matrix just for SNP accuracy computationg
    test_matrix_parts = []
    for path in test_paths:
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        end = matrix.shape[0] - (matrix.shape[0] % window_size)
        truncated_matrix = matrix[:end]
        test_matrix_parts.append(truncated_matrix)

    test_matrix = np.concatenate(test_matrix_parts, axis=0)
    test_matrix = torch.tensor(test_matrix, dtype=torch.float32, device=device)

    weights = torch.tensor(np.load(args.global_weights, allow_pickle=True), device=device)

    if args.diploid:
        final_predictions = diploid_hmm(device, test_matrix, weights)
    else:
        final_predictions = haploid_hmm(device, test_matrix, weights)

    # Read the PS4G file to get chromosome and position information
    ps4g_df = pd.read_csv(args.ps4g_file, sep="\t", comment="#")
    chroms = ps4g_df['refContig'].values
    positions = ps4g_df['refPosBinned'].values

    # Extract gamete index to name mapping from PS4G file header
    index_array = build_index_lookup(args.ps4g_file)

    bed_df = pd.DataFrame({
        "chrom": chroms[:len(final_predictions)],
        "pos": positions[:len(final_predictions)],
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })

    # Define group boundaries where parent1, parent2, or chrom changes
    group_change = (
        (bed_df["parent1"] != bed_df["parent1"].shift()) |
        (bed_df["parent2"] != bed_df["parent2"].shift()) |
        (bed_df["chrom"] != bed_df["chrom"].shift())
    )
    group_id = group_change.cumsum()

    # Collapse into ranges
    ranges_df = bed_df.groupby(group_id).agg({
        "chrom": "first",
        "pos": ["min", "max"],
        "parent1": "first",
        "parent2": "first"
    }).reset_index(drop=True)

    # Clean up MultiIndex columns
    ranges_df.columns = ["chrom", "start", "end", "parent1", "parent2"]

    # Save to BED file
    ranges_df.to_csv(args.output_bed, sep="\t", index=False)

def haploid_hmm(device, test_matrix, weights):
    log_e = F.log_softmax(test_matrix * weights, dim=-1)  # [L, num_classes]
    p_stay = 0.99
    N = log_e.shape[1]
    p_switch = (1.0 - p_stay)
    log_A = torch.full((N, N), math.log(p_switch / (N - 1)))
    log_A.fill_diagonal_(math.log(p_stay))
    log_start_probs = torch.log(torch.full((N,), 1.0 / N))

    # Viterbi decoding
    final_predictions = viterbi_decode(
        log_emit=log_e.to(device),
        log_trans=log_A.to(device),
        log_start=log_start_probs.to(device)
    )

    final_predictions = np.stack([final_predictions, final_predictions], axis=1).astype(np.int16)
    return final_predictions

def diploid_hmm(device, test_matrix, weights):
    log_e = F.log_softmax(test_matrix * weights, dim=-1)
    homo_penalty = -0.1
    N = log_e.shape[1]
    p_stay = 0.99  # tweak if needed
    p_switch = (1.0 - p_stay)
    log_A = torch.full((N, N), math.log(p_switch / (N - 1)))
    log_A.fill_diagonal_(math.log(p_stay))
    pair_states = build_pair_states(N)
    P = len(pair_states)
    log_dip_em = torch.empty(log_e.shape[0], P)

    for k, (i, j) in enumerate(pair_states):
        log_dip_em[:, k] = log_e[:, i] + log_e[:, j]
        if i == j:  # penalise homozygotes
            log_dip_em[:, k] += homo_penalty

    # diploid transition: allow **at most one chromosome to switch**
    log_dip_tr = torch.full((P, P), float('-inf'))
    for p, (a, b) in enumerate(pair_states):
        for q, (c, d) in enumerate(pair_states):
            # zero switches
            if (a, b) == (c, d):
                log_dip_tr[p, q] = log_A[a, a] + log_A[b, b]
            # one switch (a→c, b same) OR (b→d, a same)
            elif a == c and b != d:
                log_dip_tr[p, q] = log_A[a, a] + log_A[b, d]
            elif b == d and a != c:
                log_dip_tr[p, q] = log_A[a, c] + log_A[b, b]
            # two switches (disallowed / very low prob)
            else:
                log_dip_tr[p, q] = -1e6  # huge penalty

    log_start = torch.full((P,), -math.log(P))
    idx_path = viterbi_decode(log_dip_em.to(device), log_dip_tr.to(device), log_start.to(device))
    final_predictions = np.array([pair_states[i] for i in idx_path], dtype=np.int16)
    return final_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-bed", type=str, default="imputed_path.bed")
    parser.add_argument("--global-weights", type=str, default=None)
    parser.add_argument("--diploid", action="store_true")
    parser.add_argument("--ps4g-file", type=str, default=None)
    args = parser.parse_args()

    run_hmm_imputation(args)


if __name__ == '__main__':
    main()