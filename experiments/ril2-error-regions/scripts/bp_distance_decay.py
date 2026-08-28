#!/usr/bin/env python
"""
Genomic bp-DISTANCE decay of Oh43+Il14H read-support co-occurrence -- the
right test for "ambiguous reads are the expectation in a pangenome; the
problem is when identical allele sharing persists beyond where LD should
have decayed (a couple thousand bp)". Row-index-lag autocorrelation
(input_autocorrelation.py) doesn't answer this directly: row density per
Mb varies a lot with coverage, so a fixed row-lag maps to very different
bp distances at 0.01x vs 2.0x, and it conflates that with actual decay.

This computes, for TRUE (unjittered) genomic positions, the co-occurrence
correlation as a function of REAL bp separation between row pairs, bucketed
into bins fine near the 1-2kb range (where LD is expected to have mostly
decayed) and coarser further out. Uses true positions only -- this is a
property of the read data itself, not something jittering is relevant to
(jitter manipulates row ORDER for the model's benefit; this measures real
genomic distance directly).

Implementation: for each row-index offset k=1..K_MAX, computes the real bp
gap (pos[i+k]-pos[i]) and the co-occurrence product for every such pair
(vectorized per k, not per row), then buckets ALL (bp_gap, product) pairs
across all k into distance bins -- this correctly measures decay vs true
bp distance regardless of local row density, since a given k corresponds
to different bp gaps in different parts of the genome.
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

SOURCE_K = 25

# Fine near the "couple thousand bp" LD-decay scale, coarser beyond it.
BIN_EDGES = np.array([0, 150, 300, 500, 750, 1_000, 1_500, 2_000, 3_000, 5_000,
                       7_500, 10_000, 20_000, 50_000, 100_000, 500_000, 2_000_000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individual", default="Oh43xIl14H", help="IDX-RIL2 dataset dir name, e.g. Oh43xIl14H or B73xCML103")
    ap.add_argument("--parent-a", default="Oh43")
    ap.add_argument("--parent-b", default="Il14H")
    ap.add_argument("--tag", default="unfiltered-bin")
    ap.add_argument("--coverages", nargs="*", default=["0.01", "0.1", "0.5", "1.0", "2.0"])
    ap.add_argument("--k-max", type=int, default=300,
                     help="max row-index offset examined per row (pairs beyond BIN_EDGES[-1] bp discarded)")
    ap.add_argument("--json-out", required=True)
    cli = ap.parse_args()

    INDIVIDUAL = cli.individual
    PARENT_A, PARENT_B = cli.parent_a, cli.parent_b
    TAG = cli.tag

    n_bins = len(BIN_EDGES) - 1
    export = {}

    for coverage in cli.coverages:
        outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{TAG}"
        bins_path = outdir / "raw.npy.bins.tsv"
        npy_path = outdir / "raw.npy"
        if not npy_path.exists():
            print(f"SKIP {coverage}x: no {npy_path}")
            continue

        gamete_names = hae.load_gamete_names(outdir / "raw.npy.gametes.tsv")
        oh43_idx = gamete_names.index(PARENT_A)
        il14h_idx = gamete_names.index(PARENT_B)

        arr = np.load(npy_path, mmap_mode="r")
        feats = arr[:, :SOURCE_K]
        ambiguous = ((np.asarray(feats[:, oh43_idx]) != 0)
                     & (np.asarray(feats[:, il14h_idx]) != 0)).astype(np.float64)
        base_rate = ambiguous.mean()

        bins_df = pd.read_csv(bins_path, sep="\t")

        sum_xy = np.zeros(n_bins)   # sum of ambiguous[i]*ambiguous[i+k] per bin
        n_pairs = np.zeros(n_bins)  # count of pairs per bin

        for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
            idx = np.sort(idx)
            pos = bins_df.loc[idx, "bin"].to_numpy(dtype=np.int64) * 256
            amb_c = ambiguous[idx]
            n = len(idx)
            k_max_here = min(cli.k_max, n - 1)
            for k in range(1, k_max_here + 1):
                gap = pos[k:] - pos[:-k]
                # gap is monotonically non-decreasing in position but NOT
                # necessarily > BIN_EDGES[-1] just because k is large -- still
                # mask out-of-range pairs rather than assume order.
                mask = gap <= BIN_EDGES[-1]
                if not mask.any():
                    if gap.size and gap.min() > BIN_EDGES[-1]:
                        break  # all further k only increase the gap further
                    continue
                g = gap[mask]
                prod = (amb_c[k:] * amb_c[:-k])[mask]
                bin_idx = np.digitize(g, BIN_EDGES) - 1
                np.add.at(sum_xy, bin_idx, prod)
                np.add.at(n_pairs, bin_idx, 1)

        p_both = np.divide(sum_xy, n_pairs, out=np.full(n_bins, np.nan), where=n_pairs > 0)
        # P(j ambiguous | i ambiguous, distance=d) -- more interpretable than
        # raw P(both), since P(both) is mechanically capped by base_rate^2.
        p_cond = np.divide(sum_xy, n_pairs * base_rate,
                            out=np.full(n_bins, np.nan), where=n_pairs > 0)
        p_cond = np.clip(p_cond, 0, 1)

        export[coverage] = {
            "base_rate": round(float(base_rate), 4),
            "bin_edges": BIN_EDGES.tolist(),
            "n_pairs": n_pairs.astype(np.int64).tolist(),
            "p_both": [None if np.isnan(v) else round(float(v), 4) for v in p_both],
            "p_cond_given_i_ambiguous": [None if np.isnan(v) else round(float(v), 4) for v in p_cond],
        }
        print(f"\n{coverage}x: base_rate={base_rate:.4f}")
        print("bin(bp)".ljust(20) + "n_pairs".rjust(14) + "P(both)".rjust(12) + "P(j|i)".rjust(10))
        for b in range(n_bins):
            label = f"{BIN_EDGES[b]:,}-{BIN_EDGES[b+1]:,}"
            pc = export[coverage]["p_cond_given_i_ambiguous"][b]
            pb = export[coverage]["p_both"][b]
            print(f"{label:<20}{int(n_pairs[b]):>14,}{'' if pb is None else f'{pb:.4f}':>12}"
                  f"{'' if pc is None else f'{pc:.4f}':>10}")

    Path(cli.json_out).write_text(json.dumps(export, indent=1))
    print(f"\nwrote {cli.json_out}")


if __name__ == "__main__":
    main()
