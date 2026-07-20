"""
Li–Stephens HMM baseline scored on the SAME held-out test split as the CRF arms.

Reuses the emission/transition design from `hmm/hmm_impute.py:haploid_hmm`
(emission = log_softmax(reads * weight); transition = stay w.p. p_stay, else
uniform switch), but:
  * decodes each 512-window INDEPENDENTLY (the pre-windowed maize set is shuffled,
    so concatenating windows would invent transitions between unrelated samples),
  * uses a batched GPU Viterbi for speed over the 25k test windows,
  * sweeps (p_stay, weight) and reports the best, i.e. the strongest baseline.

States = the 24 founders (the read-count columns); target = the H1 label (col 24).

Usage:
    pixi run --environment gpu python src/python/crf/eval_hmm.py \
        --data /workdir/.../fullMaizeDataset_all_diploid.npy \
        --limit-n 250000 --split test
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from python.crf.train_haploid import make_splits
from python.crf.metrics import breakpoint_counts, prf

TOLS = (0, 2)


@torch.no_grad()
def batched_viterbi(log_e, log_A, log_start):
    """log_e [B,T,N], log_A [N,N], log_start [N] -> paths [B,T] (argmax states)."""
    B, T, N = log_e.shape
    dp = log_start[None] + log_e[:, 0]                  # [B,N]
    bp = torch.zeros(T, B, N, dtype=torch.long, device=log_e.device)
    for t in range(1, T):
        scores = dp.unsqueeze(2) + log_A[None]          # [B,N_prev,N_next]
        best, idx = scores.max(dim=1)                   # over prev
        dp = best + log_e[:, t]
        bp[t] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=log_e.device)
    path[:, T - 1] = dp.argmax(dim=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = bp[t].gather(1, path[:, t].unsqueeze(1)).squeeze(1)
    return path


def make_logA(N, p_stay, device):
    p_switch = 1.0 - p_stay
    log_A = torch.full((N, N), math.log(p_switch / (N - 1)), device=device)
    log_A.fill_diagonal_(math.log(p_stay))
    return log_A


@torch.no_grad()
def score(reads, labels, N, p_stay, weight, device, batch=512):
    """reads [M,T,N] int, labels [M,T] int -> (viterbi_acc, emis_acc)."""
    log_start = torch.log(torch.full((N,), 1.0 / N, device=device))
    log_A = make_logA(N, p_stay, device)
    M = reads.shape[0]
    vit_correct = emis_correct = total = 0
    bp = {t: {"tp_prec": 0, "n_pred": 0, "tp_rec": 0, "n_true": 0} for t in TOLS}
    for s in range(0, M, batch):
        rb = torch.tensor(reads[s:s + batch], dtype=torch.float32, device=device)
        lb = torch.tensor(labels[s:s + batch], dtype=torch.long, device=device)
        log_e = F.log_softmax(rb * weight, dim=-1)      # [b,T,N]
        pred_v = batched_viterbi(log_e, log_A, log_start)
        pred_e = log_e.argmax(dim=-1)
        vit_correct += (pred_v == lb).sum().item()
        emis_correct += (pred_e == lb).sum().item()
        total += lb.numel()
        for t in TOLS:
            for k, v in breakpoint_counts(pred_v, lb, tol=t).items():
                bp[t][k] += v
    return vit_correct / total, emis_correct / total, bp


def main():
    p = argparse.ArgumentParser(description="Li–Stephens HMM baseline scorer")
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--p-stay", default="0.97,0.99,0.995,0.999",
                   help="Comma list of stay probabilities to sweep.")
    p.add_argument("--weight", default="0.5,1.0,2.0",
                   help="Comma list of emission sharpness weights to sweep.")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()

    device = torch.device("cuda")
    N = args.num_parents

    _, val_ds, test_ds = make_splits(args.data, N, args.val_frac,
                                     args.test_frac, limit_n=args.limit_n)
    ds = test_ds if args.split == "test" else val_ds
    arr = ds.data                                       # [M,T,K+2]
    reads = np.ascontiguousarray(arr[:, :, :N])         # founder read counts
    labels = np.ascontiguousarray(arr[:, :, N]).astype(np.int64)  # H1

    # Empirical switch rate -> implied p_stay (context for the sweep)
    sw = (labels[:, 1:] != labels[:, :-1]).mean()
    print(f"split={args.split}  windows={len(arr):,}  "
          f"empirical switch rate={sw:.4f}  (implied p_stay≈{1 - sw:.4f})")

    p_stays = [float(x) for x in args.p_stay.split(",")]
    weights = [float(x) for x in args.weight.split(",")]
    print(f"\n{'p_stay':>8} {'weight':>8} {'viterbi':>9} {'emis_only':>10}")
    print("-" * 38)
    best = (0.0, None)
    for ps in p_stays:
        for w in weights:
            v, e, bp = score(reads, labels, N, ps, w, device)
            print(f"{ps:>8.3f} {w:>8.2f} {v:>9.4f} {e:>10.4f}")
            if v > best[0]:
                best = (v, (ps, w, e, bp))
    bv, (bps, bw, be, bbp) = best
    print(f"\nBEST HMM: p_stay={bps}  weight={bw}  "
          f"viterbi={bv:.4f}  emis_only={be:.4f}")
    bp_str = {}
    for t in TOLS:
        p, r, f1 = prf(bbp[t])
        bp_str[t] = (p, r, f1)
        print(f"  breakpoint P/R/F1 (±{t})  : {p:.3f} / {r:.3f} / {f1:.3f}  "
              f"(pred={bbp[t]['n_pred']:,} true={bbp[t]['n_true']:,})")

    res = Path(args.workdir) / "results" / "maize_eval.tsv"
    res.parent.mkdir(parents=True, exist_ok=True)
    bp_fields = "\t".join(
        f"bpP{t}={bp_str[t][0]:.3f}\tbpR{t}={bp_str[t][1]:.3f}\t"
        f"bpF{t}={bp_str[t][2]:.3f}" for t in TOLS)
    with open(res, "a") as f:
        f.write(f"hmm-LS(best p_stay={bps},w={bw})\tsplit={args.split}\t"
                f"N={len(arr) * arr.shape[1]}\tviterbi={bv:.4f}\t"
                f"emis_only={be:.4f}\t{bp_fields}\tckpt=baseline\n")
    print(f"appended → {res}")


if __name__ == "__main__":
    main()
