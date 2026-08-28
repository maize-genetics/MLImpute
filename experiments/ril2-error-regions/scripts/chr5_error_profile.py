#!/usr/bin/env python
"""
Spatial (windowed) profile of DECODE ERROR RATE across chr5 -- companion to
chr5_spatial_profile.py's input-ambiguity profile, but measuring actual
predicted-vs-true founder mismatches (reusing error_autocorrelation.py's
per_row_error_by_contig) instead of raw read-support ambiguity. Answers
"which regions of chr5 have a lot of error" directly and spatially, rather
than as one pooled genome-wide number.

Row-window based (same convention as chr5_spatial_profile.py): windows of
`window_rows` consecutive DECODED rows (i.e. after windowing's trailing-
partial-window drop), for a fixed coverage, comparing real vs 200kb-jittered
row order.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import simval_paths as P  # noqa: E402
from error_autocorrelation import per_row_error_by_contig  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
CHROM = "chr5"


def profile_from_errors(wrong, positions, window_rows):
    n_windows = len(wrong) // window_rows
    xs, ys, n = [], [], []
    for w in range(n_windows):
        sl = slice(w * window_rows, (w + 1) * window_rows)
        seg = wrong[sl]
        xs.append(int(np.median(positions[sl])))
        ys.append(float(seg.mean()))
        n.append(int(seg.sum()))
    return xs, ys, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="2.0")
    ap.add_argument("--window-rows", type=int, default=3000)
    ap.add_argument("--json-out", required=True)
    cli = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for tag, bin_size in [("unfiltered-bin", 256), ("posrand200000bp", 256)]:
        outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{cli.coverage}x__{tag}"
        errs_by_contig = per_row_error_by_contig(outdir, tag, bin_size=bin_size, device=device)
        wrong = errs_by_contig[CHROM]

        # positions: re-derive from load_contig_layout the same way
        # per_row_error_by_contig itself does, restricted to chr5 -- reuse
        # its own layout call directly instead of re-deriving bin math here.
        import heldout_assembly_eval as hae
        bins_path = outdir / "raw.npy.bins.tsv"
        T = 512  # WINDOW_SIZE, matches error_autocorrelation.py's model windowing
        layout = hae.load_contig_layout(bins_path, T, bin_size=bin_size)
        positions = None
        for contig, pos, n_windows in layout:
            if contig == CHROM:
                positions = pos
                break
        assert positions is not None and len(positions) == len(wrong), \
            f"{tag}: position/error length mismatch {len(positions) if positions is not None else None} vs {len(wrong)}"

        xs, ys, n = profile_from_errors(wrong, positions, cli.window_rows)
        results[tag] = {"pos_bp": xs, "error_rate": ys, "n_wrong": n}
        print(f"{tag}: {len(xs)} windows, overall chr5 error rate = {wrong.mean():.4%}")

    Path(cli.json_out).write_text(json.dumps({
        "chrom": CHROM, "coverage": cli.coverage, "window_rows": cli.window_rows,
        "real": results["unfiltered-bin"], "jittered": results["posrand200000bp"],
    }))
    print(f"wrote {cli.json_out}")


if __name__ == "__main__":
    main()
