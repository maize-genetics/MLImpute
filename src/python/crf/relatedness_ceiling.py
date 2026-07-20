"""
E5 headroom diagnostic: read-only ceiling vs relatedness ceiling (data-only).

The read-only IBD ceiling is 1/|S_feat|, where S_feat is the set of founders whose
read column is bit-identical to the truth across a constant-founder segment — no
per-window decoder can do better. A decoder that ALSO knows which founders the
individual descends from can discard the confusable founders the individual does
not carry, lifting the ceiling to 1/|S_feat ∩ set|, where `set` is the
individual's founder set (estimated genome-wide; here taken as ground truth from
the labels across the individual's windows). The gap between the two ceilings is
the maximum accuracy a relatedness signal could ever recover on this sim. If it is
small, individuals carry too many founders for relatedness to help; if large, the
headroom is real and a flat model result is a model/training issue, not a limit.

Data-only (no model). Usage:
    PYTHONPATH=src .pixi/envs/gpu/bin/python src/python/crf/relatedness_ceiling.py \
        --data /workdir/esb33/data/training/sim_e5_th6.npy \
        --windows-per-individual 100 --max-windows 20000
"""

import argparse
from pathlib import Path

import numpy as np

from python.crf.analyze_ibd_ceiling import BANDS


def parse_args():
    p = argparse.ArgumentParser(description="E5 read-only vs relatedness ceiling")
    p.add_argument("--data", required=True)
    p.add_argument("--ind", default="", help="default <data>.ind.npy")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--max-windows", type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    K = args.num_parents
    G = args.windows_per_individual
    data = np.load(args.data, mmap_mode="r")
    ind_path = args.ind or (Path(args.data).with_suffix("").as_posix() + ".ind.npy")
    ind = np.load(ind_path)

    n = min(args.max_windows, len(data)) if args.max_windows else len(data)
    arr = np.asarray(data[:n])
    ind = ind[:n]
    feats = arr[:, :, :K]
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K - 1)

    # Each individual's founder set = founders it ever visits (ground-truth proxy
    # for the genome-wide-estimated set).
    sets = {i: np.unique(h1[ind == i]) for i in np.unique(ind)}
    inset = np.zeros((n, K), dtype=bool)
    for i, s in sets.items():
        inset[np.ix_(np.flatnonzero(ind == i), s[s < K])] = True

    rows = np.arange(arr.shape[1])
    agg = {lbl: dict(sites=0, read=0.0, rel=0.0, drop=0) for *_, lbl in BANDS}
    agg["all"] = dict(sites=0, read=0.0, rel=0.0, drop=0)

    for w in range(n):
        h = h1[w]
        switches = int((h[1:] != h[:-1]).sum())
        lbl = next(l for lo, hi, l in BANDS if lo <= switches <= hi)
        seg_id = np.concatenate([[0], np.cumsum(h[1:] != h[:-1])])
        fe, ins = feats[w], inset[w]
        for seg in range(seg_id[-1] + 1):
            sl = rows[seg_id == seg]
            tf = h[sl[0]]
            L = sl.size
            eq = (fe[sl] == fe[sl, tf][:, None]).all(axis=0)        # [K] S_feat
            Sf = int(eq.sum())
            Sin = int((eq & ins).sum())                            # S_feat ∩ set
            for s in (agg[lbl], agg["all"]):
                s["sites"] += L
                s["read"] += L / Sf
                s["rel"] += L / max(Sin, 1)
                s["drop"] += int(Sf > Sin)

    hdr = f"{'band':>8} {'sites':>10} {'read-ceil':>10} {'rel-ceil':>9} {'headroom':>9}"
    print(f"\n[{Path(args.data).name}] read-only vs relatedness ceiling  windows={n}")
    print(hdr)
    print("-" * len(hdr))
    for lbl in [l for *_, l in BANDS] + ["all"]:
        s = agg[lbl]
        if not s["sites"]:
            continue
        rc, lc = s["read"] / s["sites"], s["rel"] / s["sites"]
        print(f"{lbl:>8} {s['sites']:>10,} {rc:>10.4f} {lc:>9.4f} {lc - rc:>+9.4f}")


if __name__ == "__main__":
    main()
