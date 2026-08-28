#!/usr/bin/env python
"""
Error-autocorrelation diagnostic -- see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md follow-up
("What does the recombination pattern look like after 100kbp jitter? ...
autocorrelation of source origin should be zero, but it will not").

Per-row correct/wrong indicator (predicted founder vs TRUE founder from
simval_truth_labels.bin_truth_labels, position-derived ground truth --
RIL2 has no per-read source label in the raw FASTQs, but is homozygous
per-locus so ground truth is exactly recoverable per PS4G/npy row), then
the sample autocorrelation of that binary sequence vs row-index lag,
per chromosome (row order = model's own view, not bp distance). Null: if
model mistakes were independent per-site noise, ACF(k) ~ 0 for k > 0;
positive ACF at short lags means errors cluster (consistent with
indel-driven local artifacts), which the recombination-overcalling result
already suggests.

Recomputes per-row (not per-collapsed-BED-interval) predictions directly
via run_inference_diploid + load_contig_layout, since write_imputed_bed
only persists the collapsed BED, not the per-row predictions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402
from simval_truth_labels import bin_truth_labels  # noqa: E402
from adaptive_drop import adaptive_drop_idx  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
PARENT_A, PARENT_B = "Oh43", "Il14H"


def per_row_error_by_contig(outdir, tag, bin_size=256, device=None):
    """Returns {contig: error_indicator (bool array, len = n_windows*T for
    that contig)} -- one entry per row actually covered by a full window
    (matches load_contig_layout's own trailing-partial-window drop)."""
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bins_path = outdir / "raw.npy.bins.tsv"
    gamete_names = hae.load_gamete_names(outdir / "raw.npy.gametes.tsv")

    # Adaptive per-sample drop (see [[git_branch_write_policy]] and
    # scripts/adaptive_drop.py's own docstring) -- hit counts are
    # order-independent, so computing this from whichever raw npy this
    # outdir has (real raw.npy for a baseline dir, raw_posrand*.npy for a
    # jitter dir) gives the identical drop_idx either way.
    raw_npy_for_hits = outdir / "raw.npy"
    if not raw_npy_for_hits.exists():
        candidates = list(outdir.glob("raw_posrand*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no raw npy found in {outdir} for adaptive drop selection")
        raw_npy_for_hits = candidates[0]
    dropped_idx, drop_name, _ = adaptive_drop_idx(raw_npy_for_hits, gamete_names)

    windowed_npy = outdir / f"windowed_k24_fixdrop{dropped_idx}_bin.npy"

    pi_arr, pj_arr, _, _ = seo.run_inference_diploid(
        windowed_npy, device, ckpt_path=Path(P.CKPT_DIPLOID), kind="ril2")
    T = pi_arr.shape[1]
    pi_flat = pi_arr.reshape(-1)
    pj_flat = pj_arr.reshape(-1)

    layout = hae.load_contig_layout(bins_path, T, bin_size=bin_size)

    bins_df = pd.read_csv(bins_path, sep="\t")
    truth_labels, _ = bin_truth_labels(
        bins_path, "ril2", gamete_names, dataset_id="IDX-RIL2",
        individual=INDIVIDUAL, parent_a=PARENT_A, parent_b=PARENT_B)
    truth_idx = truth_labels[:, 0]  # h1 == h2 for ril2

    contig_row_ranges = {
        contig: np.sort(idx)
        for contig, idx in bins_df.groupby("contig", sort=False).indices.items()
    }

    out = {}
    cursor = 0
    for contig, positions, n_windows in layout:
        n_rows = n_windows * T
        pi_c = pi_flat[cursor: cursor + n_rows]
        pj_c = pj_flat[cursor: cursor + n_rows]
        cursor += n_rows

        used_idx = contig_row_ranges[contig][: n_rows]  # same truncation load_contig_layout applies
        true_idx_c = truth_idx[used_idx]

        # Map TARGET_K-space predicted index -> SOURCE_K-space index (same inversion
        # k_target_to_name uses internally, done here on indices instead of names to
        # stay vectorized). dropped_idx computed once above, from this row's own data.
        def to_source_idx(idx_target):
            idx_target = np.asarray(idx_target)
            idx_source = idx_target.copy()
            bump = idx_target >= dropped_idx
            idx_source[bump] += 1
            return idx_source

        pred1_idx = to_source_idx(pi_c)
        pred2_idx = to_source_idx(pj_c)

        wrong = (pred1_idx != true_idx_c) | (pred2_idx != true_idx_c) | (true_idx_c < 0)
        out[contig] = wrong
    return out


def acf(x, max_lag):
    """Sample autocorrelation of a binary 0/1 array, lags 1..max_lag."""
    x = x.astype(np.float64)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def mean_run_length(x):
    """Mean length (in rows) of contiguous True/1 runs in a boolean array --
    a base-rate-robust companion to acf() (Pearson ACF of a sparse binary
    series can shift with the overall rate even at fixed absolute clustering,
    since normalization divides by a shrinking variance term). Returns
    (mean_run_length, n_runs); 0/0 if x has no True values."""
    x = np.asarray(x, dtype=bool)
    if not x.any():
        return 0.0, 0
    edges = np.diff(np.concatenate(([0], x.astype(np.int8), [0])))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    lengths = run_ends - run_starts
    return float(lengths.mean()), len(lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=["1.0", "2.0"])
    ap.add_argument("--tags", nargs="*", default=["unfiltered-bin", "posrand100kb"])
    ap.add_argument("--max-lag", type=int, default=20)
    cli = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for coverage in cli.coverages:
        for tag in cli.tags:
            outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{tag}"
            if not (outdir / "windowed_k24_fixdrop23_bin.npy").exists():
                print(f"SKIP {coverage}x/{tag}: no windowed npy")
                continue
            errs_by_contig = per_row_error_by_contig(outdir, tag, device=device)

            all_wrong = np.concatenate(list(errs_by_contig.values()))
            all_err_rate = all_wrong.mean()
            print(f"\n=== {coverage}x / {tag}  (overall per-row error rate: {all_err_rate:.4%}) ===")
            print("contig".ljust(8) + "".join(f"lag{k:<6}" for k in range(1, min(6, cli.max_lag + 1))))
            acf_by_contig = {}
            for contig, wrong in errs_by_contig.items():
                a = acf(wrong, cli.max_lag)
                acf_by_contig[contig] = a
                print(contig.ljust(8) + "".join(f"{v:<9.4f}" for v in a[:5]))
            pooled = np.mean([a for a in acf_by_contig.values()], axis=0)
            print("MEAN".ljust(8) + "".join(f"{v:<9.4f}" for v in pooled[:5]))

            # Run-length distribution of contiguous wrong-row chunks (rows,
            # not bp) -- pooled genome-wide, per-contig lengths concatenated
            # (contig boundaries never merge since each contig's array is run
            # independently through mean_run_length below).
            all_lengths = []
            for contig, wrong in errs_by_contig.items():
                edges = np.diff(np.concatenate(([0], wrong.astype(np.int8), [0])))
                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)
                all_lengths.append(ends - starts)
            all_lengths = np.concatenate(all_lengths) if all_lengths else np.array([])
            if len(all_lengths):
                pct = np.percentile(all_lengths, [10, 25, 50, 75, 90, 99])
                print(f"wrong-run length (rows): n_runs={len(all_lengths):,} mean={all_lengths.mean():.2f} "
                      f"p10={pct[0]:.0f} p25={pct[1]:.0f} median={pct[2]:.0f} p75={pct[3]:.0f} "
                      f"p90={pct[4]:.0f} p99={pct[5]:.0f} max={all_lengths.max():.0f}")


if __name__ == "__main__":
    main()
