#!/usr/bin/env python
"""
Build real intermediate-coverage read sets for Oh43xIl14H (IDX-RIL2) by
head-truncating the official 2.0x R1/R2 fastqs (wgsim reads are already
i.i.d.-random, so a prefix is a valid lower-coverage subsample -- the same
head-based-subsample convention used elsewhere in this project instead of
refmap's own --target-hits, which overshoots ~2x -- see
[[refmap_target_hits_overshoot]]), then run the baseline (unfiltered-bin,
no jitter) align stage for each, so run_ril2_posrand_sweep.py /
founder_path_error.py (neither of which consult the manifest -- they only
use the coverage string to build scratch paths) work on these coverages
unchanged. See /home/zrm22/.claude/plans/wondrous-discovering-octopus.md
follow-up (coverage vs optimal-jitter-window formula).

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_intermediate_coverages.py \
        [--coverages 0.05 0.2 0.3 0.7 1.5]
"""
import argparse
import subprocess
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

SOURCE_R1 = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/"
    "IDX-RIL2/Oh43xIl14H/Oh43xIl14H.2.0x.R1.fastq.gz")
SOURCE_R2 = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/"
    "IDX-RIL2/Oh43xIl14H/Oh43xIl14H.2.0x.R2.fastq.gz")
SOURCE_COVERAGE = 2.0
SOURCE_N_READS = 14_112_236  # wc -l/4 on the 2.0x R1 fastq, verified

SCRATCH_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval")
SUBSAMPLE_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_subsamples")
INDIVIDUAL = "Oh43xIl14H"
READS_PER_1X = SOURCE_N_READS / SOURCE_COVERAGE  # ~7,056,118


def build_subsample(coverage):
    n_reads = int(round(READS_PER_1X * coverage))
    assert n_reads <= SOURCE_N_READS, (
        f"{coverage}x needs {n_reads:,} reads > {SOURCE_N_READS:,} available in the 2.0x source")
    n_lines = n_reads * 4
    out_r1 = SUBSAMPLE_DIR / f"Oh43xIl14H.{coverage}x.R1.fastq.gz"
    out_r2 = SUBSAMPLE_DIR / f"Oh43xIl14H.{coverage}x.R2.fastq.gz"
    SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if out_r1.exists() and out_r2.exists():
        print(f"  [{coverage}x] subsample already exists, skipping ({n_reads:,} reads)")
        return out_r1, out_r2
    for src, dst in [(SOURCE_R1, out_r1), (SOURCE_R2, out_r2)]:
        cmd = f"zcat {src} | head -n {n_lines} | gzip > {dst}.tmp.gz"
        subprocess.run(cmd, shell=True, check=True)
        Path(f"{dst}.tmp.gz").rename(dst)
    print(f"  [{coverage}x] wrote {n_reads:,} reads ({n_lines:,} lines) -> {out_r1}, {out_r2}")
    return out_r1, out_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", type=float, default=[0.05, 0.2, 0.3, 0.7, 1.5])
    cli = ap.parse_args()

    hae.THREADS = "20"
    for coverage in cli.coverages:
        cov_str = str(coverage)
        print(f"\n=== IDX-RIL2 / {INDIVIDUAL} / {cov_str}x / unfiltered-bin (intermediate) ===")
        r1, r2 = build_subsample(coverage)

        outdir = SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{cov_str}x__unfiltered-bin"
        outdir.mkdir(parents=True, exist_ok=True)

        args = __import__("argparse").Namespace(
            stage="align", sample=f"{INDIVIDUAL}_ril2_unfiltered-bin",
            r1=str(r1), r2=str(r2), truth_h1=None, truth_h2=None,
            outdir=str(outdir), out_json=str(outdir / "align_result.json"),
            drop_idx=23, max_hit_frac=None, retain_counts=False,
            region=None, threads=20, no_cleanup=False,
            panel_vcf=str(seo.P.PANEL_VCF_V2), ckpt=str(seo.P.CKPT_DIPLOID),
            dataset_id="IDX-RIL2", coverage=cov_str, dataset_class="indexed",
            kind="ril2", arm="refmap", bin_size=256,
        )
        result = seo.do_align(args)
        print(f"  -> {result}")


if __name__ == "__main__":
    main()
