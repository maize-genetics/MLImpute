"""
Affinity separability probe: for each WG individual, compare the genome-wide
founder match-rate (affinity col 0) of TRUE-carried founders vs the rest. If
carried founders sit cleanly above the background, a small reduced-K' CRF can
replace the full 24-founder decode (the main speed lever).

Reports, per individual: k_true, the carried-founder rate range, the top
non-carried rate, the gap, and how many founders a threshold rule would KEEP
(recall of carried = must be 1.0 for a safe prune).
"""
import argparse
import glob as globmod
from pathlib import Path

import numpy as np

from python.crf.train_diploid import _founder_affinity


def keep_set(rate, tau_frac, floor):
    tau = max(floor, tau_frac * rate.max())
    return np.where(rate >= tau)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="/workdir/esb33/data/held-out/wg_*_dense_i0.npy")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--tau-frac", type=float, default=0.5)
    p.add_argument("--floor", type=float, default=0.0)
    args = p.parse_args()
    K = args.num_parents

    hdr = (f"{'individual':30s} {'k':>2} {'carr_lo':>7} {'carr_hi':>7} "
           f"{'noncar_hi':>9} {'gap':>6} {'keep':>4} {'recall':>6} {'P_red':>6}")
    print(hdr); print("-" * len(hdr))
    for f in sorted(globmod.glob(args.glob)):
        arr = np.load(f)
        aff = _founder_affinity(arr[:, :, :K].astype(np.float32))   # [K,2]
        rate = aff[:, 0]
        true_f = np.unique(arr[:, :, K:K + 2]).astype(int)
        true_f = true_f[true_f < K]                                  # drop null pad
        carried = np.zeros(K, bool); carried[true_f] = True
        cr = rate[carried]; nc = rate[~carried]
        keep = keep_set(rate, args.tau_frac, args.floor)
        recall = carried[keep].sum() / max(1, carried.sum())
        kp = len(keep) + 1                                           # +null founder
        p_red = kp * (kp + 1) // 2
        name = Path(f).stem
        print(f"{name:30s} {len(true_f):>2d} {cr.min():>7.3f} {cr.max():>7.3f} "
              f"{nc.max():>9.3f} {cr.min()-nc.max():>6.3f} {len(keep):>4d} "
              f"{recall:>6.2f} {p_red:>6d}")


if __name__ == "__main__":
    main()
