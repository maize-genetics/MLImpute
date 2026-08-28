#!/usr/bin/env python
"""
Reconstruct a real PS4G v2.0 file for a jitter-window arm, for validation in
external tools (e.g. the grits app) -- run_ril2_posrand_sweep.py only ever
writes the jittered raw_posrand{bp}bp.npy + bins.tsv sidecar pair (all the
downstream pipeline needs), never a .ps4g text file.

Faithful reconstruction relies on two confirmed facts about raw.npy:
  1. It stores real per-founder READ COUNTS (not binarized -- binarization
     only happens later, at the windowing step's --retain-counts flag), so
     no count information is lost between raw.npy and a re-derived .ps4g.
  2. Within any one row, every nonzero founder column holds the SAME value
     -- that shared value IS the PS4G row's single `count` field (verified
     across 200k real rows, zero exceptions). So gameteSet = nonzero column
     indices (already 0-indexed matching PS4G's own gamete indices), count
     = that row's shared nonzero value.

Jittering never touches read counts or gameteSet composition, only which
bin each row is reported at -- so the header's per-founder total counts
are identical to the source (real) raw.ps4g and are copied verbatim; only
the `#Command:` line and the data rows' refPosBinned differ.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import simval_paths as P  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
SOURCE_K = 25
SOURCE_TAG = "unfiltered-bin"


def write_jittered_ps4g(jit_dir, source_ps4g_path, out_path):
    bins_path = jit_dir / "raw.npy.bins.tsv"
    npy_candidates = list(jit_dir.glob("raw_posrand*.npy"))
    if not npy_candidates:
        raise FileNotFoundError(f"no jittered raw_posrand*.npy in {jit_dir}")
    npy_path = npy_candidates[0]

    bins_df = pd.read_csv(bins_path, sep="\t")
    arr = np.load(npy_path, mmap_mode="r")
    feats = np.asarray(arr[:, :SOURCE_K])

    # Header: copy verbatim from the source (real) ps4g up to and including
    # the per-gamete total-count block, since none of that depends on row
    # order -- only swap the #Command: line for traceability.
    header_lines = []
    with open(source_ps4g_path) as f:
        for line in f:
            if line.startswith("#Command:"):
                header_lines.append(
                    f"#Command: [reconstructed by write_jittered_ps4g.py from "
                    f"{npy_path.name} -- positions jittered within a fixed window "
                    f"from the real refmap output at {source_ps4g_path}]\n")
                continue
            header_lines.append(line)
            if line.startswith("gameteSet\t"):
                break

    with open(out_path, "w") as out:
        out.writelines(header_lines)
        contigs = bins_df["contig"].to_numpy()
        bins = bins_df["bin"].to_numpy()
        for i in range(len(bins_df)):
            row = feats[i]
            nz = np.nonzero(row)[0]
            if len(nz) == 0:
                continue
            count = int(row[nz[0]])
            gameteset_str = ",".join(str(int(x)) for x in nz)
            out.write(f"{gameteset_str}\t{contigs[i]}\t{int(bins[i])}\t{count}\n")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=["0.01", "0.1", "0.5", "1.0", "2.0"])
    ap.add_argument("--tag", default="posrand200000bp")
    cli = ap.parse_args()

    for coverage in cli.coverages:
        jit_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{cli.tag}"
        source_ps4g = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{SOURCE_TAG}" / "raw.ps4g"
        out_path = jit_dir / "raw_jittered.ps4g"
        if not jit_dir.exists():
            print(f"SKIP {coverage}x: no {jit_dir}")
            continue
        if not source_ps4g.exists():
            print(f"SKIP {coverage}x: no source ps4g at {source_ps4g}")
            continue
        write_jittered_ps4g(jit_dir, source_ps4g, out_path)
        n_lines = sum(1 for _ in open(out_path)) - 30  # minus header
        print(f"{coverage}x: wrote {out_path}  (~{n_lines:,} data rows)")


if __name__ == "__main__":
    main()
