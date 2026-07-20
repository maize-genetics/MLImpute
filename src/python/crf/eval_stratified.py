"""
Stratified comparison: founder accuracy + breakpoint P/R on the test windows,
restricted to a band of true-breakpoint counts per window (default 0–2, the
"easy" low-recombination windows).

Compares the four headline arms on the IDENTICAL filtered subset:
  crf-full (per-site c), crf-d128L4, crf-windowc, HMM Li–Stephens.

Usage:
    pixi run --environment gpu python src/python/crf/eval_stratified.py \
        --data /workdir/.../fullMaizeDataset_all_diploid.npy \
        --limit-n 250000 --lo 0 --hi 2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from python.crf.train_haploid import (
    GRITSCRFHaploid, PreWindowedHaploidDataset, make_splits)
from python.crf.metrics import breakpoint_counts, prf
from python.crf.eval_hmm import batched_viterbi, make_logA
from torch.utils.data import DataLoader

TOLS = (0, 2)
CKPT_ROOT = "/workdir/esb33/checkpoints"
ARMS = ["crf-full", "crf-d128L4", "crf-windowc"]   # CRF arms (resolve best ckpt)


def best_ckpt(run_name):
    cks = sorted(Path(CKPT_ROOT, run_name).glob("*/*.ckpt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cks[0]) if cks else None


@torch.no_grad()
def eval_crf(ckpt, subset, device, bs=128):
    model = GRITSCRFHaploid.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    ds = PreWindowedHaploidDataset(subset, num_parents=24)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    correct = n = 0
    bp = {t: {"tp_prec": 0, "n_pred": 0, "tp_rec": 0, "n_true": 0} for t in TOLS}
    for b in loader:
        X = b["input_embeds"].to(device); tags = b["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X)
        pred = model.decode(emis_f, c)
        correct += (pred == tags).sum().item(); n += tags.numel()
        for t in TOLS:
            for k, v in breakpoint_counts(pred, tags, tol=t).items():
                bp[t][k] += v
    return correct / n, bp


@torch.no_grad()
def eval_hmm(subset, N, p_stay, weight, device, bs=512):
    reads = np.ascontiguousarray(subset[:, :, :N])
    labels = np.ascontiguousarray(subset[:, :, N]).astype(np.int64)
    log_start = torch.log(torch.full((N,), 1.0 / N, device=device))
    log_A = make_logA(N, p_stay, device)
    correct = n = 0
    bp = {t: {"tp_prec": 0, "n_pred": 0, "tp_rec": 0, "n_true": 0} for t in TOLS}
    for s in range(0, len(reads), bs):
        rb = torch.tensor(reads[s:s + bs], dtype=torch.float32, device=device)
        lb = torch.tensor(labels[s:s + bs], dtype=torch.long, device=device)
        log_e = F.log_softmax(rb * weight, dim=-1)
        pred = batched_viterbi(log_e, log_A, log_start)
        correct += (pred == lb).sum().item(); n += lb.numel()
        for t in TOLS:
            for k, v in breakpoint_counts(pred, lb, tol=t).items():
                bp[t][k] += v
    return correct / n, bp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--lo", type=int, default=0, help="min true breakpoints/window")
    p.add_argument("--hi", type=int, default=2, help="max true breakpoints/window")
    p.add_argument("--hmm-p-stay", type=float, default=0.995)
    p.add_argument("--hmm-weight", type=float, default=0.5)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()

    device = torch.device("cuda")
    N = args.num_parents
    _, _, test_ds = make_splits(args.data, N, args.val_frac, args.test_frac,
                                limit_n=args.limit_n)
    arr = test_ds.data
    labels = np.asarray(arr[:, :, N], dtype=np.int64)
    nbp = (labels[:, 1:] != labels[:, :-1]).sum(axis=1)            # per window
    mask = (nbp >= args.lo) & (nbp <= args.hi)
    subset = np.ascontiguousarray(arr[mask])
    frac = mask.mean()
    print(f"Stratum: true breakpoints in [{args.lo},{args.hi}]  "
          f"windows={mask.sum():,}/{len(arr):,} ({frac*100:.1f}%)  "
          f"sites={subset.shape[0]*subset.shape[1]:,}")
    print(f"(full-test mean bp/window = {nbp.mean():.2f}, median = "
          f"{int(np.median(nbp))})\n")

    rows = []
    for arm in ARMS:
        ck = best_ckpt(arm)
        if ck is None:
            print(f"  (skip {arm}: no checkpoint)"); continue
        acc, bp = eval_crf(ck, subset, device)
        rows.append((arm, acc, bp))
    acc, bp = eval_hmm(subset, N, args.hmm_p_stay, args.hmm_weight, device)
    rows.append((f"hmm-LS(p{args.hmm_p_stay}/w{args.hmm_weight})", acc, bp))

    print(f"{'model':<26}{'acc':>8}{'bpP±2':>8}{'bpR±2':>8}{'bpF1±2':>8}"
          f"{'pred_bp':>10}")
    print("-" * 68)
    for arm, acc, bp in rows:
        p2, r2, f2 = prf(bp[2])
        print(f"{arm:<26}{acc:>8.4f}{p2:>8.3f}{r2:>8.3f}{f2:>8.3f}"
              f"{bp[2]['n_pred']:>10,}")

    res = Path(args.workdir) / "results" / "maize_stratified.tsv"
    res.parent.mkdir(parents=True, exist_ok=True)
    with open(res, "a") as f:
        for arm, acc, bp in rows:
            p0, r0, f0 = prf(bp[0]); p2, r2, f2 = prf(bp[2])
            f.write(f"bp[{args.lo},{args.hi}]\t{arm}\twindows={int(mask.sum())}\t"
                    f"acc={acc:.4f}\tbpF1_0={f0:.3f}\tbpP2={p2:.3f}\t"
                    f"bpR2={r2:.3f}\tbpF1_2={f2:.3f}\n")
    print(f"\nappended → {res}")


if __name__ == "__main__":
    main()
