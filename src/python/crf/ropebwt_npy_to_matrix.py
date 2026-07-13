"""
Convert a ropebwt3-phg PS4G/numpy export (`refmap --npy=... --ps4g=... --label-bed=...`,
branch `worktree-refmap-ps4g-numpy` of maize-genetics/ropebwt3-phg) into the standard
grits training-matrix contract consumed by `crf/eval.py` / `crf/train_haploid.py` /
`crf/train_diploid.py` (`make_splits` / `make_diploid_splits`).

The two formats differ in three load-bearing ways (not just a dtype cast):
  1. dtype: ropebwt3-phg writes int32; grits expects int8 (read-counts clipped to [0,127],
     matching the convention already used by `cross/build_training_data.py`).
  2. shape: ropebwt3-phg's `.npy` is FLAT — one row per (contig, bin, gameteSet), with a
     companion `<npy>.bins.tsv` (row -> contig, bin) and `<npy>.gametes.tsv` (column -> founder
     name). grits expects PRE-WINDOWED `[N, window_size, K+2]` blocks. This script groups rows
     by contig (bins.tsv is already sorted contig-then-bin) and slices consecutive
     `window_size`-row chunks, matching the same non-collapsed, row-not-position windowing
     convention the external `extract_windows_single.py` scripts use for the maize/cassava
     training sets, dropping a trailing partial window per contig.
  3. unknown-label sentinel: ropebwt3-phg writes `-1` for unlabeled bins (a standard,
     tool-agnostic missing-value convention). grits' `PreWindowedHaploidDataset`/`make_splits`
     silently mis-handles `-1` as founder-index 0 (`np.clip(labels, 0, K)`), NOT as the CRF's
     real "unknown" state (index `K`) — so this script remaps `-1 -> K` explicitly.

This fix is deliberately kept downstream (here), not pushed into ropebwt3-phg's exporter: the
windowing size and the unknown-state encoding are grits-CRF-specific choices, and even grits
itself keeps windowing external to its own PS4G parser.

Usage:
    python src/python/crf/ropebwt_npy_to_matrix.py \
        --npy ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy \
        --bins ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.bins.tsv \
        --gametes ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.gametes.tsv \
        --num-parents 25 --window-size 512 \
        --out grits_workdir/data/real/ropebwt_oh43_1M.npy

To point at new reads later, only --npy/--bins/--gametes (and --out) need to change.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="ropebwt3-phg PS4G/npy -> grits training matrix")
    p.add_argument("--npy", required=True,
                   help="ropebwt3-phg --npy export (int32, [n_rows, K+2]).")
    p.add_argument("--bins", required=True,
                   help="Companion <npy>.bins.tsv (row, contig, bin).")
    p.add_argument("--gametes", default=None,
                   help="Companion <npy>.gametes.tsv (gameteIndex, sampleName). Optional; "
                        "used only to sanity-check --num-parents and log founder order.")
    p.add_argument("--num-parents", type=int, default=None,
                   help="K. Inferred from --gametes or npy.shape[1]-2 if not given.")
    p.add_argument("--window-size", type=int, default=512,
                   help="Rows per window (matches grits' T=512 site-window convention).")
    p.add_argument("--step-size", type=int, default=None,
                   help="Window stride; defaults to --window-size (non-overlapping).")
    p.add_argument("--out", required=True, help="Output .npy path.")
    return p.parse_args()


def main():
    args = parse_args()
    window_size = args.window_size
    step_size = args.step_size or window_size

    arr = np.load(args.npy, mmap_mode="r")
    bins_df = pd.read_csv(args.bins, sep="\t")
    assert list(bins_df.columns) == ["row", "contig", "bin"], \
        f"unexpected bins.tsv columns: {list(bins_df.columns)}"
    assert len(bins_df) == arr.shape[0], (
        f"row-count mismatch: npy has {arr.shape[0]} rows, bins.tsv has {len(bins_df)}")
    assert np.array_equal(bins_df["row"].to_numpy(), np.arange(len(bins_df))), \
        "bins.tsv 'row' column is not a plain 0..N-1 sequence — cannot align by position"

    gametes = None
    if args.gametes:
        gametes = pd.read_csv(args.gametes, sep="\t")
        assert list(gametes.columns) == ["gameteIndex", "sampleName"], \
            f"unexpected gametes.tsv columns: {list(gametes.columns)}"

    K = args.num_parents
    if K is None:
        K = len(gametes) if gametes is not None else arr.shape[1] - 2
    if gametes is not None:
        assert len(gametes) == K, (
            f"--num-parents {K} disagrees with {len(gametes)} founders in --gametes")
    assert arr.shape[1] == K + 2, (
        f"npy has {arr.shape[1]} cols; expected K+2={K + 2} for K={K}")

    arr = np.asarray(arr)                                     # materialize the mmap
    feats = np.clip(arr[:, :K], 0, 127).astype(np.int8)
    gA = arr[:, K].astype(np.int64)
    gB = arr[:, K + 1].astype(np.int64)
    unk = gA < 0
    gA[unk] = K                                                # -1 -> real "unknown" state
    unk = gB < 0
    gB[unk] = K
    labels = np.stack([gA, gB], axis=1).astype(np.int8)        # [n_rows, 2]
    rows = np.concatenate([feats, labels], axis=1)             # [n_rows, K+2] int8

    windows = []
    contig_window_counts = {}
    for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)                                     # preserve on-disk order
        n_windows = 0
        start = 0
        while start + window_size <= len(idx):
            sel = idx[start:start + window_size]
            windows.append(rows[sel])
            n_windows += 1
            start += step_size
        contig_window_counts[contig] = n_windows

    if not windows:
        raise ValueError("no complete windows produced — window_size too large for the data")
    out = np.stack(windows, axis=0)                            # [N, window_size, K+2] int8

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)

    het = (out[:, :, K] != out[:, :, K + 1])
    labeled = (out[:, :, K] != K) | (out[:, :, K + 1] != K)
    print(f"\nWrote {out_path}")
    print(f"  shape={out.shape}  dtype={out.dtype}  size={out.nbytes / 1e9:.3f} GB")
    print(f"  source rows={arr.shape[0]:,}  windows={out.shape[0]:,}  "
          f"window_size={window_size}  step_size={step_size}")
    nonzero = {c: n for c, n in contig_window_counts.items() if n > 0}
    n_empty = len(contig_window_counts) - len(nonzero)
    print(f"  contigs             : {len(contig_window_counts)} seen, "
          f"{len(nonzero)} contributed windows, {n_empty} too sparse (<{window_size} rows)")
    print(f"    " + ", ".join(f"{c}={n}" for c, n in sorted(nonzero.items())))
    print(f"  label coverage      : {labeled.mean()*100:.1f}%  "
          f"(unknown state = index {K})")
    print(f"  het (gA != gB)      : {het.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
