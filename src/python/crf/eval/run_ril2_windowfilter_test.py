#!/usr/bin/env python
"""
One-off validation runner for the new max-hit-frac / binarize windowing
features (see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md).

The edited ropebwt_npy_to_matrix.py lives on a feature branch/worktree
(grits-windowfilter-worktree, branch windowing-quality-filters off
origin/tripsacum-tests) that isn't merged yet, so this script monkey-patches
nam_baseline.WINDOW_SCRIPT to point there for this run only -- same
in-process module-global-patch pattern simval_eval_one.py already uses for
the v1->v2 FMD/LIFT cutover. No shared file is repointed permanently.

Reuses the already-built raw.npy/raw.ps4g for IDX-RIL2__Oh43xIl14H__0.1x
(run_refmap is resumable, so this reruns only windowing + inference).

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_windowfilter_test.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT
print(f"[patched] hae.nb.WINDOW_SCRIPT -> {WORKTREE_WINDOW_SCRIPT}")

BASE_ROW = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval/"
    "IDX-RIL2__Oh43xIl14H__0.1x")

R1 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_datasets/IDX-RIL2/Oh43xIl14H/Oh43xIl14H.0.1x.R1.fastq.gz"
R2 = "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_datasets/IDX-RIL2/Oh43xIl14H/Oh43xIl14H.0.1x.R2.fastq.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hit-frac", type=float, default=0.5)
    cli = ap.parse_args()
    tag = f"hitfrac{cli.max_hit_frac}-bin"
    outdir = Path(
        "/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/simval_eval/"
        f"IDX-RIL2__Oh43xIl14H__0.1x__{tag}")

    outdir.mkdir(parents=True, exist_ok=True)
    # reuse the already-built raw.npy/raw.ps4g/raw.tsv from the baseline row --
    # run_refmap() only looks at outdir for npy_path/tsv_path, so symlinking
    # them into the new outdir makes it skip refmap entirely (resumable check).
    for name in ("raw.npy", "raw.npy.bins.tsv", "raw.npy.gametes.tsv", "raw.ps4g",
                 "raw.tsv", "raw.log"):
        src = BASE_ROW / name
        dst = outdir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    args = argparse.Namespace(
        stage="align", sample=f"Oh43xIl14H_ril2_{tag}",
        r1=R1, r2=R2, truth_h1=None, truth_h2=None,
        outdir=str(outdir), out_json=str(outdir / "align_result.json"),
        drop_idx=23, max_hit_frac=cli.max_hit_frac, retain_counts=False,
        region=None, threads=20, no_cleanup=False,
        panel_vcf=str(seo.P.PANEL_VCF_V2), ckpt=str(seo.P.CKPT_DIPLOID),
        dataset_id="IDX-RIL2", coverage="0.1", dataset_class="indexed", kind="ril2",
        arm="refmap",
    )
    hae.THREADS = str(args.threads)
    result = seo.do_align(args)
    print("\n=== align_result ===")
    print(result)


if __name__ == "__main__":
    main()
