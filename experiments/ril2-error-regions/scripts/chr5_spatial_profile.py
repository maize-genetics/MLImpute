#!/usr/bin/env python
"""
Spatial (windowed) profile of the Oh43+Il14H read-support co-occurrence
autocorrelation across chr5 -- extends input_autocorrelation.py's single
genome-wide-pooled ACF(lag=1) into a per-locus track, to see whether chr5's
disproportionate share of clustering (consistently ~16-18% of genome-wide
total at every jitter window size, see
results/ril2_binsize_posrand_diagnostics.md) comes from one concentrated
region or is diffuse across the whole chromosome.

CORRECTED SIGNAL (2026-08-25, user-flagged): earlier this used fanout>1
(matches ANY >1 of 25 founders) -- wrong, since multi-founder read support
is the routine, expected outcome of pangenome alignment against founders
that share sequence by descent (fires on ~91-93% of the genome, not a
meaningful signal). Now uses the same decision-relevant definition as the
Su1-locus and chr5:189-202Mb spot checks: does the row's read support fail
to distinguish specifically between Oh43 and Il14H, this individual's own
two true parents.

For a fixed row-window size (default 3000 consecutive rows -- large enough
for a stable lag-1 Pearson ACF estimate, small enough for real spatial
resolution), computes per window: co-occurrence density, local ACF(lag=1),
and the window's genomic midpoint (bp) -- for BOTH real (position-sorted)
and 200kb-jittered row order, at a chosen coverage.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_paths as P  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
PARENT_A, PARENT_B = "Oh43", "Il14H"
SOURCE_K = 25
CHROM = "chr5"


def windowed_profile(bins_path, npy_path, gametes_path, window_rows):
    bins_df = pd.read_csv(bins_path, sep="\t")
    idx = bins_df.index[bins_df["contig"] == CHROM].to_numpy()
    idx = np.sort(idx)
    arr = np.load(npy_path, mmap_mode="r")
    feats = arr[idx, :SOURCE_K]
    gamete_names = hae.load_gamete_names(gametes_path)
    oh43_idx = gamete_names.index(PARENT_A)
    il14h_idx = gamete_names.index(PARENT_B)
    ambiguous = ((np.asarray(feats[:, oh43_idx]) != 0)
                 & (np.asarray(feats[:, il14h_idx]) != 0)).astype(np.float64)
    bp = bins_df.loc[idx, "bin"].to_numpy(dtype=np.int64) * 256

    n_windows = len(idx) // window_rows
    positions, density, local_acf1 = [], [], []
    for w in range(n_windows):
        sl = slice(w * window_rows, (w + 1) * window_rows)
        seg = ambiguous[sl]
        positions.append(int(np.median(bp[sl])))
        density.append(float(seg.mean()))
        x = seg - seg.mean()
        denom = np.dot(x, x)
        local_acf1.append(float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else 0.0)
    return positions, density, local_acf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="2.0")
    ap.add_argument("--window-rows", type=int, default=3000)
    ap.add_argument("--json-out", required=True)
    cli = ap.parse_args()

    real_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{cli.coverage}x__unfiltered-bin"
    jit_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{cli.coverage}x__posrand200000bp"

    real_pos, real_dens, real_acf = windowed_profile(
        real_dir / "raw.npy.bins.tsv", real_dir / "raw.npy",
        real_dir / "raw.npy.gametes.tsv", cli.window_rows)
    jit_npy = list(jit_dir.glob("raw_posrand*.npy"))[0]
    jit_pos, jit_dens, jit_acf = windowed_profile(
        jit_dir / "raw.npy.bins.tsv", jit_npy,
        real_dir / "raw.npy.gametes.tsv", cli.window_rows)

    out = {
        "chrom": CHROM, "coverage": cli.coverage, "window_rows": cli.window_rows,
        "real": {"pos_bp": real_pos, "density": real_dens, "acf1": real_acf},
        "jittered": {"pos_bp": jit_pos, "density": jit_dens, "acf1": jit_acf},
    }
    Path(cli.json_out).write_text(json.dumps(out))
    print(f"wrote {cli.json_out}  (real: {len(real_pos)} windows, jittered: {len(jit_pos)} windows)")


if __name__ == "__main__":
    main()
