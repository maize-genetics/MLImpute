#!/usr/bin/env python
"""
Build the `binsize1` tag for IDX-RIL2 B73xCML103 at 0.5x -- no prior
binsize1 run exists for this pair (only Oh43xIl14H had one already, built
earlier this session). Requires a genuine fresh refmap run (bin-size
changes the C-level PS4G row-collapsing, so raw.npy differs from the
`unfiltered-bin` 256bp-bin run -- unlike --max-hit-frac, which is a
downstream windowing-only filter and can reuse an existing raw.npy via
run_ril2_all_pairs.py's stage_raw_from_sibling trick).

Uses the corpus's own official reads (manifest.tsv r1/r2), matching
run_ril2_all_pairs.py's documented preference over the ad hoc
scratch/read_datasets build (which that script's docstring notes does NOT
byte-match the official fastqs). drop_idx=23 (P39) confirmed via
adaptive_drop_idx to be the genuinely correct (not just historically
hardcoded) per-sample adaptive drop for this pair -- P39 has the lowest
hit count of all 25 founders here (1,545,457, vs B73/CML103's 3.4M/3.0M as
the true parents) -- so this matches both the old convention and the
adaptive-drop fix.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT

MANIFEST = Path("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
INDIVIDUAL = "B73xCML103"
COVERAGE = "0.5"
TAG = "binsize1"
DROP_IDX = 23  # P39 -- confirmed correct adaptive drop for this pair, see docstring


def load_manifest_row(individual, coverage):
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] == "IDX-RIL2" and row["individual"] == individual \
                    and row["coverage"] == coverage:
                return row
    raise KeyError(f"no manifest row for IDX-RIL2/{individual}/{coverage}x")


def main():
    row = load_manifest_row(INDIVIDUAL, COVERAGE)
    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{COVERAGE}x__{TAG}"
    outdir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        sample=f"{INDIVIDUAL}_ril2_{TAG}",
        r1=row["r1_path"], r2=row["r2_path"],
        outdir=str(outdir),
        arm="refmap",
        bin_size=1,
        no_cleanup=False,
        drop_idx=DROP_IDX,
        max_hit_frac=None,
        retain_counts=False,
        kind="ril2",
        ckpt=str(P.CKPT_DIPLOID),
    )
    print(f"r1={args.r1}\nr2={args.r2}\noutdir={outdir}")
    info = seo.do_align(args)
    print(info)


if __name__ == "__main__":
    main()
