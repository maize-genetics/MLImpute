#!/usr/bin/env python
"""
Jitter-noise-floor control for the 100kbp position-randomization probe --
see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md. Since the
posrand100kb sweep uses the jittered position for scoring too, even a
PERFECT decode would show nonzero founder-path bp-weighted error near true
crossover breakpoints (coordinate smearing of up to +-100kb). This computes
that floor directly: build a "decoded" BED from the true mosaic's own
breakpoints jittered the identical way (randomize_positions_100kb.
jitter_mosaic_positions), with founder calls exactly right, then score it
against the TRUE (unjittered) mosaic with founder_path_error's proven
metric -- no model, no refmap, seconds to run.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from simval_oracle_bed import build_ril_mosaics  # noqa: E402
from randomize_positions_100kb import jitter_mosaic_positions  # noqa: E402
from founder_path_error import score_founder_path_error  # noqa: E402

TMP_BED_DIR = Path("/home/zrm22/.claude/jobs/0e12e67f/tmp/posrand_control_bed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-a", required=True)
    ap.add_argument("--parent-b", required=True)
    ap.add_argument("--window-bp", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", args.parent_a, args.parent_b)
    jittered = jitter_mosaic_positions(mosaic_h1, window_bp=args.window_bp, seed=args.seed)

    sample = "jitter_control"
    bed_dir = TMP_BED_DIR
    bed_dir.mkdir(parents=True, exist_ok=True)
    for chrom, segs in jittered.items():
        rows = [{"chrom": chrom, "start": s, "end": e,
                 "parent1": label_to_name[lab], "parent2": label_to_name[lab]}
                for s, e, lab in segs]
        pd.DataFrame(rows)[["chrom", "start", "end", "parent1", "parent2"]].to_csv(
            bed_dir / f"{sample}_{chrom}_imputed.bed", sep="\t", index=False)

    wrong_bp, total_bp, error_pct = score_founder_path_error(
        bed_dir, sample, mosaic_h1, label_to_name)
    print(f"jitter-noise-floor (window_bp={args.window_bp}, seed={args.seed}): "
          f"wrong_bp={wrong_bp:,} total_bp={total_bp:,} error_pct={error_pct:.6f}%")


if __name__ == "__main__":
    main()
