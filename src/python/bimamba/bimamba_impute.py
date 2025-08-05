import numpy as np
import pandas as pd
import torch
import argparse
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from python.bimamba.bimamba_model import BiMambaSmooth
from python.ps4g_io.torch_loaders import WindowIndexDataset


def decode_position(encoded_pos):
    """
    Decode a 32-bit integer that packs:
      • the upper-8 bits → an index (0-255)
      • the lower-24 bits → a position, but quantised in bins of 256 bp
    This is lossy because we multiplied by 256 during encoding.
    """
    idx = (encoded_pos >> 24) & 0xFF           # top-byte index (unsigned)
    pos = (encoded_pos & 0x0FFFFFF) * 256      # restore to bp units
    return idx, pos

def build_pair_states(n_parents):
    """Return list [(i,j) …] with i ≤ j covering all unordered parent pairs."""
    return [(i, j) for i in range(n_parents) for j in range(i, n_parents)]

def make_diploid_emissions(log_em, pair_states):
    """
    Diploid emission = sum of two haploid log-probs (independent chromosomes).
    log_em:  T × N
    returns: T × P  where P=len(pair_states)
    """
    T, _ = log_em.shape
    dip = torch.empty(T, len(pair_states), dtype=log_em.dtype, device=log_em.device)
    for k, (i, j) in enumerate(pair_states):
        dip[:, k] = log_em[:, i] + log_em[:, j]
    return dip

def make_diploid_transitions(log_A, pair_states):
    """
    Diploid transition under independent chromosome movement:
      P((a,b)→(c,d)) = A[a,c] * A[b,d]   (unordered c≤d)
    log_A : N × N
    returns: P × P log-prob matrix
    """
    P = len(pair_states)
    dip = torch.empty(P, P, dtype=log_A.dtype, device=log_A.device)
    for p, (a, b) in enumerate(pair_states):
        outer = log_A[a].unsqueeze(1) + log_A[b]          # N × N
        for q, (c, d) in enumerate(pair_states):
            dip[p, q] = outer[c, d]
    return dip

def viterbi_decode(log_emit, log_trans, log_start):
    """
    Generic log-space Viterbi for a discrete HMM.
    log_emit : T × S
    log_trans: S × S
    log_start: S
    returns   : list[int] length T (best-path state indices)
    """
    T, S = log_emit.shape
    dp = torch.full((T, S), float('-inf'), dtype=log_emit.dtype, device=log_emit.device)
    bp = torch.zeros((T, S), dtype=torch.long, device=log_emit.device)

    dp[0] = log_start + log_emit[0]
    for t in range(1, T):
        scores = dp[t - 1].unsqueeze(1) + log_trans  # S × S
        dp[t], bp[t] = torch.max(scores, dim=0)      # best prev → state j
        dp[t] += log_emit[t]

    # backtrack
    state = torch.argmax(dp[-1]).item()
    path = [state]
    for t in range(T - 1, 0, -1):
        state = bp[t, state].item()
        path.append(state)
    path.reverse()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-bed", type=str, default="imputed_path.bed")
    parser.add_argument("--global-weights", type=str, default=None)
    parser.add_argument("--HMM", type=bool, default=False)
    parser.add_argument("--diploid", type=bool, default=False)
    parser.add_argument("--ps4g-file", type=str, default=None)
    args = parser.parse_args()

    window_size = 512
    num_classes = 25
    batch_size = 64
    d_model = 128
    num_layers = 3
    num_features = 25
    step_size = window_size
    lambda_smooth = 0.2

    model = BiMambaSmooth(input_dim=num_features, d_model=d_model, num_classes=num_classes, n_layer=num_layers, lambda_smooth=lambda_smooth)
    model_checkpoint = "bimamba_model.pth"
    model.load_state_dict(torch.load(model_checkpoint))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load in test matrix
    test_paths = [args.input_path]

    test_dataset = WindowIndexDataset(test_paths, window_size=window_size, top_n=num_classes,
                                      step_size=step_size, return_decode=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Reconstruct test_matrix just for SNP accuracy computation
    test_matrix_parts = []
    for path in test_paths:
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        end = matrix.shape[0] - (matrix.shape[0] % window_size)
        truncated_matrix = matrix[:end]
        test_matrix_parts.append(truncated_matrix)

    test_matrix = np.concatenate(test_matrix_parts, axis=0)
    test_matrix = torch.tensor(test_matrix, dtype=torch.float32, device=device)

    model.eval()

    if not args.diploid and not args.HMM:  # haploid ML only
        final_predictions = []
        with torch.no_grad():
            for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
                batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)  # decode_dict: [B, top_n]
                outputs, mask = model(batch_data)
                batch_predictions = torch.argmax(outputs, dim=-1)  # [B, L]
                B, L = batch_predictions.shape  # batch size and window size
                pred_labels = batch_predictions.reshape(-1)  # [B*L]
                row_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)  # [B*L]
                final_predictions.append(decode_dict[row_ids, pred_labels])  # [B*L]
        final_predictions = torch.cat(final_predictions).cpu().numpy()  # shape (N,)
        final_predictions = np.stack([final_predictions, final_predictions], axis=1)

    elif args.diploid and not args.HMM: # diploid ML only
        final_predictions = []
        with torch.no_grad():
            for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
                batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)  # decode_dict: [B, top_n]
                outputs, mask = model(batch_data)
                probs = torch.sigmoid(outputs)  # [B, L, num_classes]
                top2_probs, top2_parents = torch.topk(probs, k=2, dim=-1)  # [B, L, 2]
                ratio = top2_probs[..., 1] / top2_probs[..., 0].clamp(min=1e-8)  # [B, L]
                is_diploid = ratio > 0.8

                B, L, _ = top2_parents.shape
                row_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, L)  # [B, L]
                parent1 = decode_dict[row_ids, top2_parents[..., 0]]  # [B, L]
                parent2 = decode_dict[row_ids, top2_parents[..., 1]]  # [B, L]

                # Build prediction tuples based on diploid flag
                pred = torch.stack([
                    parent1,
                    torch.where(is_diploid, parent2, parent1)  # use parent2 if diploid, else repeat parent1
                ], dim=-1)  # [B, L, 2]

                # Convert to list of lists of tuples: [[(p1, p2), (p1, p2), ...], ...]
                batch_predictions = [
                    [tuple(pair.tolist()) for pair in sample]  # sample: [L, 2]
                    for sample in pred
                ]
                final_predictions.extend(batch_predictions)
        final_predictions = [tup for batch in final_predictions for tup in batch]  # flatten
        final_predictions = np.array(final_predictions, dtype=np.int16)  # shape (N, 2)

    elif args.diploid and args.HMM: # diploid ML + HMM
        final_logits = []
        decode_dicts = []
        with torch.no_grad():
            for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
                batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)  # decode_dict: [B, top_n]
                outputs, mask = model(batch_data)
                final_logits.append(outputs)
                decode_dicts.append(decode_dict)
        decode_dict_full = torch.cat(decode_dicts, dim=0)
        logits_concat = torch.cat(final_logits, dim=0)  # Shape: [T, 512, 25]
        flattened = logits_concat.view(-1, num_classes)  # Shape: [T * 512, 25]
        truncated = flattened[:, :test_matrix.shape[1]]
        log_e = F.log_softmax(truncated, dim=-1)

        weights = np.load(args.global_weights, allow_pickle=True)['weights']
        homo_penalty = -0.1
        N = log_e.shape[1]
        p_stay = float(weights.max()) * 0.20  # tweak if needed
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
        final_predictions = np.array([
            (
                decode_dict_full[i // window_size, p1].item(),
                decode_dict_full[i // window_size, p2].item()
            )
            for i, (p1, p2) in enumerate(final_predictions)
        ], dtype=np.int16)

    else: # haploid ML + HMM
        final_logits = []
        decode_dicts = []
        with torch.no_grad():
            for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
                batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)  # decode_dict: [B, top_n]
                outputs, mask = model(batch_data)
                final_logits.append(outputs)
                decode_dicts.append(decode_dict)
        decode_dict_full = torch.cat(decode_dicts, dim=0)
        logits_concat = torch.cat(final_logits, dim=0)  # Shape: [T, 512, 25]
        flattened = logits_concat.view(-1, num_classes)  # Shape: [T * 512, 25]
        truncated = flattened[:, :test_matrix.shape[1]]
        log_e = F.log_softmax(truncated, dim=-1)

        weights = np.load(args.global_weights, allow_pickle=True)['weights']
        N = log_e.shape[1]
        p_stay = float(weights.max()) * 0.20  # tweak if needed
        p_switch = (1.0 - p_stay)
        log_A = torch.full((N, N), math.log(p_switch / (N - 1)))
        log_A.fill_diagonal_(math.log(p_stay))

        log_start_probs = torch.log(torch.full((N,), 1.0 / N))

        final_predictions = viterbi_decode(
            log_emit=log_e.to(device),
            log_trans=log_A.to(device),
            log_start=log_start_probs.to(device)
        )
        final_predictions = np.stack([final_predictions, final_predictions], axis=1).astype(np.int16)
        final_predictions = np.array([
            (
                decode_dict_full[i // window_size, p1].item(),
                decode_dict_full[i // window_size, p2].item()
            )
            for i, (p1, p2) in enumerate(final_predictions)
        ], dtype=np.int16)

    spline_pos = pd.read_csv(args.ps4g_file, sep="\t", comment="#")['pos']
    decoded = np.vstack(np.vectorize(decode_position)(spline_pos)).T
    chroms, positions = zip(*decoded)

    with open(args.ps4g_file, 'r') as file:
        comments = [line for line in file if line.startswith('#')]

    gamete_data = []

    for line in comments:
        line = line.strip()
        if line.startswith("#") and ":" in line and "\t" in line:
            # Example line: "#B73:0\t1\t10730006"
            line = line[1:]  # Remove leading "#"
            gamete_full, idx, count = line.split("\t")
            gamete_name = gamete_full.split(":")[0]
            gamete_data.append({
                "gamete": gamete_name,
                "gamete_index": int(idx),
            })

    index_to_name = {entry["gamete_index"]: entry["gamete"] for entry in gamete_data}
    max_index = max(index_to_name.keys())  # ensure all indices fit
    index_array = [index_to_name[i] for i in range(max_index + 1)]

    bed_df = pd.DataFrame({
        # TODO: convert chr_idx to chr
        "chrom_idx": chroms[:len(final_predictions)],
        "pos": positions[:len(final_predictions)],
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })

    # Save to BED file
    bed_df.to_csv(args.output_bed, sep="\t", index=False)

if __name__ == '__main__':
    main()