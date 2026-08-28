#!/usr/bin/env python
"""
Combined arm: --bin-size=1 refmap output + the same 100kbp position-jitter
transform applied on top -- see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md ("RIL2
indel-density diagnostics"). User-requested addition: run both probes
TOGETHER, not just as two independent arms, to see whether they compound.

Sources each coverage's ALREADY-COMPLETE binsize1 raw.npy/raw.npy.bins.tsv
(from run_ril2_binsize_sweep.py) and jitters with bin_size=1 (matching the
source data's own resolution -- jitter_positions' bin_size param must match
whatever produced the input bins.tsv, same requirement as bin_size=256 for
the unfiltered-bin source). Skips any coverage whose binsize1 row isn't
done yet -- safe to re-invoke as more coverages land.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_binsize1_posrand_sweep.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402
from randomize_positions_100kb import jitter_positions  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT
print(f"[patched] hae.nb.WINDOW_SCRIPT -> {WORKTREE_WINDOW_SCRIPT}")

SCRATCH_ROOT = P.SCRATCH_ROOT
INDIVIDUAL = "Oh43xIl14H"
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]
SOURCE_TAG = "binsize1"
SOURCE_BIN_SIZE = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=COVERAGES)
    ap.add_argument("--individual", default=INDIVIDUAL)
    ap.add_argument("--window-bp", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()
    tag = "binsize1_posrand100kb"

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for coverage in cli.coverages:
        src_dir = SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{coverage}x__{SOURCE_TAG}"
        raw_npy_path = src_dir / "raw.npy"
        bins_path = src_dir / "raw.npy.bins.tsv"
        if not raw_npy_path.exists() or not bins_path.exists():
            print(f"  SKIP {coverage}x: binsize1 row not done yet ({raw_npy_path})")
            continue

        outdir = SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{coverage}x__{tag}"
        bed_dir = outdir / "bed"
        if bed_dir.exists() and len(list(bed_dir.glob("*_imputed.bed"))) >= 10:
            print(f"  SKIP {coverage}x: {tag} already scored")
            continue

        print(f"\n=== IDX-RIL2 / {cli.individual} / {coverage}x / {tag} ===")
        outdir.mkdir(parents=True, exist_ok=True)

        gametes_dst = outdir / "raw.npy.gametes.tsv"
        if not gametes_dst.exists():
            shutil.copy(src_dir / "raw.npy.gametes.tsv", gametes_dst)

        jittered_npy_path = outdir / f"raw_{tag}.npy"
        jittered_bins_path = outdir / "raw.npy.bins.tsv"  # hae.window()'s fixed lookup path
        if not jittered_npy_path.exists() or not jittered_bins_path.exists():
            arr = np.load(raw_npy_path)
            bins_df = pd.read_csv(bins_path, sep="\t")
            new_bins_df, new_arr = jitter_positions(
                bins_df, arr, window_bp=cli.window_bp, bin_size=SOURCE_BIN_SIZE, seed=cli.seed)
            assert new_arr.shape == arr.shape
            assert len(new_bins_df) == len(bins_df)
            np.save(jittered_npy_path, new_arr)
            new_bins_df.to_csv(jittered_bins_path, sep="\t", index=False)
            print(f"  wrote jittered raw npy/bins ({len(new_bins_df):,} rows, "
                  f"window_bp={cli.window_bp}, bin_size={SOURCE_BIN_SIZE}, seed={cli.seed})")

        windowed_npy, wf_bins_path = seo.window_fixed_drop(
            jittered_npy_path, outdir, drop_idx=23, max_hit_frac=None, retain_counts=False)

        gamete_names = hae.load_gamete_names(gametes_dst)
        pi_arr, pj_arr, het_scale_val, het_scale_diag = seo.run_inference_diploid(
            windowed_npy, device, ckpt_path=Path(P.CKPT_DIPLOID), kind="ril2")

        hae.write_imputed_bed(f"{cli.individual}_ril2_{tag}", pi_arr, pj_arr, 23,
                               gamete_names, wf_bins_path, bed_dir, bin_size=SOURCE_BIN_SIZE)
        print(f"  -> wrote BED to {bed_dir}")


if __name__ == "__main__":
    main()
