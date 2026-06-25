"""
Li–Stephens diploid HMM baseline scored on the same held-out test split as
the diploid CRF arms.

Emission model: log_softmax(reads * weight) per founder, combined as
  log_e_pair[k] = log_e[i] + log_e[j] - homo_penalty * (i == j)
Transition: independent per-chromosome stay/switch (p_stay), so a two-
chromosome switch costs (p_switch)^2.  Same factored form as train_diploid.py
but with a fixed c rather than a learned one.

Decodes each 512-window independently (the pre-windowed set is shuffled).
Sweeps (p_stay, weight, homo_penalty) and reports the best Viterbi pair_acc.

Usage:
    pixi run --environment gpu python src/python/crf/eval_diploid_hmm.py \
        --data /workdir/esb33/data/training/sim_diploid_512.npy \
        --limit-n 100000 --split test
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from python.crf.train_diploid import build_pair_tables, PreWindowedDiploidDataset
from python.crf.train_haploid import make_splits


@torch.no_grad()
def batched_viterbi_diploid(log_e_pair, log_tr, log_start):
    """
    log_e_pair  [B,T,P]  pair-state emission log-probs
    log_tr      [P,P]    transition log-probs (fixed across time)
    log_start   [P]
    returns paths [B,T] of pair-state indices
    """
    B, T, P = log_e_pair.shape
    dp = log_start[None] + log_e_pair[:, 0]        # [B,P]
    bp = torch.zeros(T, B, P, dtype=torch.long, device=log_e_pair.device)
    for t in range(1, T):
        # [B,P_prev,P_next]
        scores = dp.unsqueeze(2) + log_tr[None]    # broadcast over batch
        best, idx = scores.max(dim=1)              # best prev state
        dp = best + log_e_pair[:, t]
        bp[t] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=log_e_pair.device)
    path[:, T - 1] = dp.argmax(dim=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t].gather(1, path[:, t + 1].unsqueeze(1)).squeeze(1)
    return path


def build_diploid_logtr(K, p_stay, device):
    """
    Factored per-chromosome transition: independent stay/switch.
    log_tr[p,q] = log P(h1 switch) + log P(h2 switch) where each chromosome
    independently switches with p_switch = 1 - p_stay.
    """
    _, _, _, nsw = build_pair_tables(K)              # [P,P] float {0,1,2}
    nsw = nsw.to(device)
    p_switch = 1.0 - p_stay
    log_stay = math.log(p_stay)
    log_sw = math.log(p_switch / (K - 1))
    # 0 switches -> 2*log_stay; 1 switch -> log_stay + log_sw; 2 -> 2*log_sw
    log_tr = (2 - nsw) * log_stay + nsw * log_sw
    return log_tr                                    # [P,P]


@torch.no_grad()
def score(reads, h1_labels, h2_labels, pair_labels, K, p_stay, weight,
          homo_penalty, device, batch=256):
    """
    reads        [M,T,K]  float read counts
    h1_labels    [M,T]    founder index
    h2_labels    [M,T]
    pair_labels  [M,T]    pair state index (from build_pair_tables)
    Returns (pair_acc, hap_acc).
    """
    pi, pj, _, _ = build_pair_tables(K)
    pi = pi.to(device)
    pj = pj.to(device)
    P = pi.shape[0]

    log_tr = build_diploid_logtr(K, p_stay, device)
    log_start = torch.full((P,), -math.log(P), device=device)

    M = reads.shape[0]
    pair_correct = hap_correct = total_pairs = total_hap = 0

    for s in range(0, M, batch):
        rb = torch.tensor(reads[s:s + batch], dtype=torch.float32, device=device)
        h1b = torch.tensor(h1_labels[s:s + batch], dtype=torch.long, device=device)
        h2b = torch.tensor(h2_labels[s:s + batch], dtype=torch.long, device=device)
        plb = torch.tensor(pair_labels[s:s + batch], dtype=torch.long, device=device)

        log_e = F.log_softmax(rb * weight, dim=-1)   # [B,T,K]
        # pair emission: log_e[i] + log_e[j], penalise homozyg
        log_e_pair = log_e[:, :, pi] + log_e[:, :, pj]  # [B,T,P]
        homozyg = (pi == pj).float().to(device)
        log_e_pair = log_e_pair - homo_penalty * homozyg[None, None]

        pred_p = batched_viterbi_diploid(log_e_pair, log_tr, log_start)  # [B,T]

        pair_correct += (pred_p == plb).sum().item()
        total_pairs += plb.numel()

        # per-haplotype: both pred and truth sorted (lo<=hi), matching
        # train_diploid.py:_accuracy so the numbers are directly comparable.
        pred_lo = pi[pred_p]                         # [B,T]
        pred_hi = pj[pred_p]
        t_lo = torch.minimum(h1b, h2b)
        t_hi = torch.maximum(h1b, h2b)
        hap_correct += (
            (pred_lo == t_lo).sum() + (pred_hi == t_hi).sum()
        ).item()
        total_hap += t_lo.numel() + t_hi.numel()

    return pair_correct / total_pairs, hap_correct / total_hap


def main():
    p = argparse.ArgumentParser(description="Diploid HMM baseline scorer")
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--p-stay", default="0.97,0.99,0.995,0.999")
    p.add_argument("--weight", default="0.5,1.0,2.0")
    p.add_argument("--homo-penalty", default="0.5,1.0,2.0,3.0",
                   help="Homozygous pair log-prob penalty to sweep.")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()

    device = torch.device("cuda")
    K = args.num_parents

    _, val_ds, test_ds = make_splits(
        args.data, K, args.val_frac, args.test_frac, limit_n=args.limit_n)
    ds = test_ds if args.split == "test" else val_ds
    arr = ds.data                                    # [M,T,K+2]

    reads = np.ascontiguousarray(arr[:, :, :K]).astype(np.float32)
    h1 = np.ascontiguousarray(arr[:, :, K]).astype(np.int64)
    h2 = np.ascontiguousarray(arr[:, :, K + 1]).astype(np.int64)

    # Build pair label matrix from (h1,h2)
    _, _, pair_table, _ = build_pair_tables(K)
    pair_table_np = pair_table.numpy()
    pair_labels = np.array([
        pair_table_np[np.minimum(h1[i], h2[i]), np.maximum(h1[i], h2[i])]
        for i in range(len(h1))
    ], dtype=np.int64)

    sw_h1 = (h1[:, 1:] != h1[:, :-1]).mean()
    sw_h2 = (h2[:, 1:] != h2[:, :-1]).mean()
    print(f"split={args.split}  windows={len(arr):,}  "
          f"H1 switch rate={sw_h1:.4f}  H2 switch rate={sw_h2:.4f}")

    p_stays = [float(x) for x in args.p_stay.split(",")]
    weights = [float(x) for x in args.weight.split(",")]
    homo_pens = [float(x) for x in args.homo_penalty.split(",")]

    print(f"\n{'p_stay':>8} {'weight':>8} {'homo_pen':>10} "
          f"{'pair_acc':>10} {'hap_acc':>9}")
    print("-" * 52)

    best = (0.0, None)
    for ps in p_stays:
        for w in weights:
            for hp in homo_pens:
                pa, ha = score(reads, h1, h2, pair_labels, K, ps, w, hp, device)
                print(f"{ps:>8.3f} {w:>8.2f} {hp:>10.2f} {pa:>10.4f} {ha:>9.4f}")
                if pa > best[0]:
                    best = (pa, (ps, w, hp, ha))

    bpa, (bps, bw, bhp, bha) = best
    print(f"\nBEST diploid HMM:  p_stay={bps}  weight={bw}  "
          f"homo_penalty={bhp}  pair_acc={bpa:.4f}  hap_acc={bha:.4f}")

    res = Path(args.workdir) / "results" / "diploid_eval.tsv"
    res.parent.mkdir(parents=True, exist_ok=True)
    with open(res, "a") as f:
        f.write(
            f"hmm-LS-diploid(p_stay={bps},w={bw},hp={bhp})\t"
            f"split={args.split}\tN={len(arr)}\t"
            f"pair_acc={bpa:.4f}\thap_acc={bha:.4f}\tckpt=baseline\n"
        )
    print(f"appended → {res}")


if __name__ == "__main__":
    main()
