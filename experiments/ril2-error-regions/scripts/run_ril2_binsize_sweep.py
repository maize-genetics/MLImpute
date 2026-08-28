#!/usr/bin/env python
"""
IDX-RIL2 Oh43xIl14H, all 5 manifest coverages, refmap run with
--bin-size=1 instead of the default 256 -- see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md ("RIL2
indel-density diagnostics"). Tests whether refmap's 256bp position
quantization (interacting with real-genome indel-driven local read-density
irregularities) is contributing to the founder-path error that rises with
coverage in the existing unfiltered-bin grid.

Deliberately unfiltered (no --max-hit-frac) -- user wants these new probes
scored against the unfiltered baseline for scalability, matching the
existing IDX-RIL2__Oh43xIl14H__{cov}x__unfiltered-bin rows exactly except
for bin_size.

Each row gets its OWN fresh outdir (tag "binsize1") and genuinely re-runs
refmap -- unlike run_ril2_all_pairs.py's stage_raw_from_sibling shortcut,
raw.npy content is fundamentally different per bin-size (a bin-size=256
raw.npy symlinked under a binsize1 tag would silently produce wrong
results), so that shortcut is intentionally NOT used here.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_binsize_sweep.py \
        [--coverages 0.01 0.1 0.5 1.0 2.0] [--bin-size 1]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT
print(f"[patched] hae.nb.WINDOW_SCRIPT -> {WORKTREE_WINDOW_SCRIPT}")

MANIFEST = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
SCRATCH_ROOT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval")
INDIVIDUAL = "Oh43xIl14H"
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]


def load_manifest_row(individual, coverage):
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] == "IDX-RIL2" and row["individual"] == individual \
                    and row["coverage"] == coverage:
                return row
    raise KeyError(f"no manifest row for IDX-RIL2/{individual}/{coverage}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=COVERAGES)
    ap.add_argument("--bin-size", type=int, default=1)
    ap.add_argument("--individual", default=INDIVIDUAL)
    cli = ap.parse_args()
    tag = f"binsize{cli.bin_size}"

    hae.THREADS = "20"
    for coverage in cli.coverages:
        mrow = load_manifest_row(cli.individual, coverage)
        outdir = SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{coverage}x__{tag}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== IDX-RIL2 / {cli.individual} / {coverage}x / {tag} ===")

        args = argparse.Namespace(
            stage="align", sample=f"{cli.individual}_ril2_{tag}",
            r1=mrow["r1_path"], r2=mrow["r2_path"], truth_h1=None, truth_h2=None,
            outdir=str(outdir), out_json=str(outdir / "align_result.json"),
            drop_idx=23, max_hit_frac=None, retain_counts=False,
            region=None, threads=20, no_cleanup=False,
            panel_vcf=str(seo.P.PANEL_VCF_V2), ckpt=str(seo.P.CKPT_DIPLOID),
            dataset_id="IDX-RIL2", coverage=coverage, dataset_class="indexed",
            kind="ril2", arm="refmap", bin_size=cli.bin_size,
        )
        result = seo.do_align(args)
        print(f"  -> {result}")


if __name__ == "__main__":
    main()
