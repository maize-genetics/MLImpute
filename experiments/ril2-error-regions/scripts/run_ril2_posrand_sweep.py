#!/usr/bin/env python
"""
IDX-RIL2 Oh43xIl14H, all 5 manifest coverages, 100kbp position-jitter probe
-- see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md ("RIL2
indel-density diagnostics"). Reuses each coverage's ALREADY-COMPLETE
unfiltered-bin raw.npy/raw.npy.bins.tsv (no refmap re-run needed -- this is
a pure post-hoc reorder of already-good data, see randomize_positions_100kb
.jitter_positions), then re-derives a windowed_k25 -> windowed_k24 ->
inference -> BED path directly, bypassing do_align/prep_fastq/run_refmap
entirely (unnecessary here since there are no fresh reads to align).

hae.window()'s bins_path is a FIXED outdir/"raw.npy.bins.tsv" regardless of
which npy path is passed in, so the jittered bins.tsv is written there
under a fresh outdir (tag "posrand100kb") and the jittered npy is passed
explicitly to window_fixed_drop -- no symlink/short-circuit tricks needed.

Deliberately unfiltered (no --max-hit-frac), matching every other row in
this diagnostic (see run_ril2_binsize_sweep.py's docstring for why).

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_posrand_sweep.py \
        [--coverages 0.01 0.1 0.5 1.0 2.0] [--seed 0]
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
from adaptive_drop import adaptive_drop_idx  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT
print(f"[patched] hae.nb.WINDOW_SCRIPT -> {WORKTREE_WINDOW_SCRIPT}")

SCRATCH_ROOT = P.SCRATCH_ROOT
INDIVIDUAL = "Oh43xIl14H"
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]
SOURCE_TAG = "unfiltered-bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=COVERAGES)
    ap.add_argument("--individual", default=INDIVIDUAL)
    ap.add_argument("--window-bp", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()
    # Collision-proof: encode the exact bp value, not a truncated/rounded
    # unit label (1_500_000 // 1_000_000 == 1 would otherwise silently
    # collide with the real 1_000_000 tag and overwrite its directory).
    tag = f"posrand{cli.window_bp}bp"

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for coverage in cli.coverages:
        print(f"\n=== IDX-RIL2 / {cli.individual} / {coverage}x / {tag} ===")
        src_dir = SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{coverage}x__{SOURCE_TAG}"
        raw_npy_path = src_dir / "raw.npy"
        bins_path = src_dir / "raw.npy.bins.tsv"
        if not raw_npy_path.exists() or not bins_path.exists():
            print(f"  SKIP: missing {raw_npy_path} or {bins_path}")
            continue

        outdir = SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{coverage}x__{tag}"
        outdir.mkdir(parents=True, exist_ok=True)

        # gamete names are unchanged by jitter (same columns, just reordered
        # rows) -- reuse the source dir's sidecar directly.
        gametes_dst = outdir / "raw.npy.gametes.tsv"
        if not gametes_dst.exists():
            shutil.copy(src_dir / "raw.npy.gametes.tsv", gametes_dst)

        jittered_npy_path = outdir / f"raw_{tag}.npy"
        jittered_bins_path = outdir / "raw.npy.bins.tsv"  # hae.window()'s fixed lookup path
        if not jittered_npy_path.exists() or not jittered_bins_path.exists():
            arr = np.load(raw_npy_path)
            bins_df = pd.read_csv(bins_path, sep="\t")
            new_bins_df, new_arr = jitter_positions(
                bins_df, arr, window_bp=cli.window_bp, bin_size=256, seed=cli.seed)
            assert new_arr.shape == arr.shape
            assert len(new_bins_df) == len(bins_df)
            np.save(jittered_npy_path, new_arr)
            new_bins_df.to_csv(jittered_bins_path, sep="\t", index=False)
            print(f"  wrote jittered raw npy/bins ({len(new_bins_df):,} rows, "
                  f"window_bp={cli.window_bp}, seed={cli.seed})")

        gamete_names = hae.load_gamete_names(gametes_dst)
        # Hit counts are order-independent, so the adaptive drop_idx computed
        # from the RAW (pre-jitter) npy is identical to what jittered data
        # would give -- compute once per (sample, coverage), reuse across
        # every jitter-window arm for a fair, apples-to-apples comparison.
        drop_idx, drop_name, _ = adaptive_drop_idx(raw_npy_path, gamete_names)
        print(f"  adaptive drop: {drop_name} (idx={drop_idx})")

        windowed_npy, wf_bins_path = seo.window_fixed_drop(
            jittered_npy_path, outdir, drop_idx=drop_idx, max_hit_frac=None, retain_counts=False)

        pi_arr, pj_arr, het_scale_val, het_scale_diag = seo.run_inference_diploid(
            windowed_npy, device, ckpt_path=Path(P.CKPT_DIPLOID), kind="ril2")

        bed_dir = outdir / "bed"
        hae.write_imputed_bed(f"{cli.individual}_ril2_{tag}", pi_arr, pj_arr, drop_idx,
                               gamete_names, wf_bins_path, bed_dir, bin_size=256)
        print(f"  -> wrote BED to {bed_dir}")


if __name__ == "__main__":
    main()
