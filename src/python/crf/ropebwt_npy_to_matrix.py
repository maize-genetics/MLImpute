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

Optional: `--target-num-parents` trims a real panel down to a checkpoint's trained `num_parents`
(e.g. a real K=25 panel -> K=24 to match a checkpoint trained on 24 simulated founders, no
retraining needed). Drops the least-covered founder(s) genome-wide — "least covered" = fewest
rows with any nonzero read support for that founder — until the panel reaches the target size.
Any true label that pointed at a dropped founder becomes unknown (state `K`), same as an
unlabeled bin; those instances are genuinely unrepresentable in the smaller panel.

Optional: `--max-hit-frac` drops rows whose read-group is close to uninformative. A row here is
one (contig, bin, gameteSet) group (see point 2 above) — its nonzero-column count IS the exact
founder fan-out of the reads that produced it (ropebwt3-phg writes one npy row per gameteSet by
design, specifically so this co-occurrence/fan-out information isn't collapsed away). A read
matching most of the founder panel doesn't discriminate between them; `--max-occ` (the refmap
flag) does NOT filter this -- it caps total pangenome occurrence count, not founder diversity, so
a read hitting all K founders once each survives it untouched. `--max-hit-frac 0.5` drops any row
whose fan-out exceeds half of the SOURCE panel size (computed before `--target-num-parents`, so
the threshold means the same thing regardless of what a caller later trims down to). Row-dropping
shifts every downstream window boundary (windowing below is row-count-based, not position-based),
so `<out>.bins.tsv` is written alongside `--out` with the same schema as `--bins`, reflecting only
the surviving rows -- callers that independently re-derive window/contig layout from a bins.tsv
(e.g. heldout_assembly_eval.load_contig_layout) must use this file, not the original `--bins`,
whenever `--max-hit-frac` is set, or genomic coordinates will desync from the actual windows here.

Optional: `--retain-counts` keeps raw read counts (the historical default) instead of binarizing
to 0/1 presence. Binarizing is now the DEFAULT: every checkpoint this converter feeds
(`diploid-affinity-sim512-h3` and friends) was trained exclusively on binary 0/1 features
(`simulate_alleles.py` builds strictly binary match matrices) -- feeding raw counts (real data
sees values up to 15-70) is a genuine train/eval mismatch, not a cosmetic choice: the encoder's
`cell(log1p(X))` path and the recombination-cost `depth = log1p(X.sum(-1))` feature both treat X
as a continuous magnitude, and `_founder_affinity`/`_het_scale` (train_diploid.py) both assume
bounded 0/1 input and silently misbehave on counts (an unbounded mean / an inflated Jaccard).

Usage:
    python src/python/crf/ropebwt_npy_to_matrix.py \
        --npy ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy \
        --bins ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.bins.tsv \
        --gametes ropebwt_refMap/bench_ps4g_npy/results_fixed/full_1M.npy.gametes.tsv \
        --num-parents 25 --window-size 512 \
        --out grits_workdir/data/real/ropebwt_oh43_1M.npy
        [--target-num-parents 24]

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
    p.add_argument("--target-num-parents", type=int, default=None,
                   help="If given and < the source panel size, drop the least-covered "
                        "founder(s) genome-wide (fewest rows with any nonzero read support) "
                        "until the panel reaches this size. Labels pointing at a dropped "
                        "founder become unknown. Default: off, panel size unchanged.")
    p.add_argument("--max-hit-frac", type=float, default=None,
                   help="Drop rows (read-groups) whose gameteSet fan-out exceeds this fraction "
                        "of the SOURCE panel size K (e.g. 0.5 = drop rows hitting >12/25 "
                        "founders) -- near-uninformative reads that match too much of the panel "
                        "to be diagnostic. Default: off, no fan-out filtering.")
    p.add_argument("--retain-counts", action="store_true",
                   help="Keep raw read counts instead of binarizing to 0/1 presence. Default: "
                        "off -- features are binarized, matching every checkpoint's binary "
                        "training data (see module docstring).")
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
    gA[gA < 0] = K                                             # -1 -> real "unknown" state
    gB[gB < 0] = K

    if args.max_hit_frac is not None:
        fanout = (feats != 0).sum(axis=1)                      # [n_rows] == this row's |gameteSet|
        max_founders = max(1, int(args.max_hit_frac * K))      # K here is still the SOURCE panel size
        keep = fanout <= max_founders
        n_drop = int((~keep).sum())
        print(f"--max-hit-frac {args.max_hit_frac}: dropping {n_drop:,}/{len(keep):,} rows "
              f"({n_drop / len(keep) * 100:.1f}%) with fan-out > {max_founders} founders "
              f"(of source K={K})")
        feats = feats[keep]
        gA = gA[keep]
        gB = gB[keep]
        bins_df = bins_df[keep].reset_index(drop=True)
        bins_df["row"] = np.arange(len(bins_df))               # re-sequence to match filtered rows

        bins_out_path = Path(str(args.out) + ".bins.tsv")
        bins_out_path.parent.mkdir(parents=True, exist_ok=True)
        bins_df.to_csv(bins_out_path, sep="\t", index=False)
        print(f"  wrote filtered bins sidecar -> {bins_out_path} ({len(bins_df):,} rows)")

    if not args.retain_counts:
        feats = (feats != 0).astype(np.int8)

    if args.target_num_parents is not None:
        target = args.target_num_parents
        if target > K:
            raise ValueError(f"--target-num-parents {target} > source panel size {K}")
        if target < K:
            hits = (feats != 0).sum(axis=0)                    # [K] rows-with-coverage, genome-wide
            n_drop = K - target
            drop_idx = np.argsort(hits, kind="stable")[:n_drop]
            keep_idx = np.setdiff1d(np.arange(K), drop_idx)    # sorted ascending
            names = (gametes.sort_values("gameteIndex")["sampleName"].to_numpy()
                     if gametes is not None else np.array([str(i) for i in range(K)]))
            print(f"Dropping {n_drop} least-covered founder(s) (target K={target}, source K={K}):")
            for i in np.sort(drop_idx):
                print(f"  {names[i]:<12} idx={i:<3} hits={hits[i]:,}")
            print(f"  lowest kept: {names[keep_idx[np.argmin(hits[keep_idx])]]}  "
                  f"hits={hits[keep_idx].min():,}")

            remap = np.full(K + 1, target, dtype=np.int64)     # default: dropped + old unknown -> new unknown
            remap[keep_idx] = np.arange(len(keep_idx))         # kept founders -> compacted index
            gA = remap[gA]
            gB = remap[gB]
            feats = feats[:, keep_idx]
            K = target
        else:
            print(f"--target-num-parents {target} == source panel size; nothing to drop")

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
