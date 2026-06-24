"""
E2 evaluation: does the encoder's inferred transition cost c_t track the HIDDEN
recombination rate it was never told?

The simulator (simulate_alleles.py --recomb-span>1) places breakpoints by a hidden
per-site rate map and stores that true rate in the trailing column (K+2) for
evaluation only — it is not a model input. This script runs the trained encoder on
held-out windows, pulls per-site c_t, and correlates it with the hidden rate.

Expectation: high rate ⇒ cheap to switch ⇒ LOW c_t, so a strong *negative*
Spearman correlation indicates the encoder learned to infer local recombination
from the allele-match patterns (LD) alone.

Usage:
    pixi run --environment gpu -- python src/python/crf/eval_recomb.py \
        --ckpt <path>.ckpt --data /workdir/esb33/data/training/sim_alleles_e2.npy
"""

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from python.crf.train_haploid import GRITSCRFHaploid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--n-windows", type=int, default=2000,
                   help="Held-out windows (taken from the tail) to evaluate")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    K = args.num_parents
    data = np.load(args.data, mmap_mode="r")
    if data.shape[-1] != K + 3:
        raise ValueError(f"{args.data} has no recomb track (cols={data.shape[-1]}, "
                         f"need K+3={K + 3}); regenerate with --recomb-span>1")
    data = np.asarray(data[-args.n_windows:])           # tail = held-out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    feats = torch.tensor(data[:, :, :K], dtype=torch.float32)
    rate = data[:, :, K + 2].astype(np.float64)         # hidden true rate
    h1 = data[:, :, K].astype(np.int64)
    switch = np.zeros_like(rate)
    switch[:, 1:] = (h1[:, 1:] != h1[:, :-1])

    c_all = []
    with torch.no_grad():
        for s in range(0, len(feats), args.batch_size):
            xb = feats[s:s + args.batch_size].to(device)
            _, _, c = model(xb)                          # c: [B, T], no rate fed in
            c_all.append(c.float().cpu().numpy())
    c = np.concatenate(c_all, 0)                         # [N, T]

    # Per-site correlation of inferred c_t vs hidden rate (drop site 0: no transition)
    cf, rf, sf = c[:, 1:].ravel(), rate[:, 1:].ravel(), switch[:, 1:].ravel()
    rho_rate, _ = spearmanr(cf, rf)
    rho_sw, _ = spearmanr(cf, sf)

    # Does c_t separate hot from cold? Compare c in top vs bottom rate decile.
    lo, hi = np.percentile(rf, 10), np.percentile(rf, 90)
    c_cold, c_hot = cf[rf <= lo].mean(), cf[rf >= hi].mean()

    print(f"\nEval on {len(data):,} held-out windows  (c_t inferred from features only)")
    print(f"  c_t range            : min={cf.min():.3f}  mean={cf.mean():.3f}  "
          f"max={cf.max():.3f}  sd={cf.std():.3f}")
    print(f"  Spearman(c_t, rate)  : {rho_rate:+.4f}  (expect NEGATIVE: hot ⇒ low cost)")
    print(f"  Spearman(c_t, switch): {rho_sw:+.4f}  (expect NEGATIVE at true breakpoints)")
    print(f"  mean c_t  cold decile: {c_cold:.3f}   hot decile: {c_hot:.3f}   "
          f"(cold-hot = {c_cold - c_hot:+.3f})")


if __name__ == "__main__":
    main()
