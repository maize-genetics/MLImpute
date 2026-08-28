#!/usr/bin/env python
"""
Rebuild the Oh43xIl14H (IDX-RIL2) bin-size=1 sweep using the correct
per-sample adaptive founder-drop (adaptive_drop.adaptive_drop_idx) -- same
fix as rebuild_ril2_baseline_adaptive_drop.py, applied here since
run_ril2_binsize_sweep.py had the identical hardcoded drop_idx=23 (P39)
bug, predating the fix. raw.npy already exists for every coverage (refmap
doesn't need to rerun), so this bypasses do_align/prep_fastq/run_refmap and
calls window_fixed_drop -> inference -> write_imputed_bed directly with
bin_size=1, overwriting the old, wrong-drop BED files in place.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/rebuild_ril2_binsize1_adaptive_drop.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402
from adaptive_drop import adaptive_drop_idx  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT

INDIVIDUAL = "Oh43xIl14H"
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]
TAG = "binsize1"
BIN_SIZE = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=COVERAGES)
    cli = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for coverage in cli.coverages:
        outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{TAG}"
        raw_npy = outdir / "raw.npy"
        if not raw_npy.exists():
            print(f"SKIP {coverage}x: no raw.npy at {raw_npy}")
            continue

        gamete_names = hae.load_gamete_names(outdir / "raw.npy.gametes.tsv")
        drop_idx, drop_name, hits = adaptive_drop_idx(raw_npy, gamete_names)
        print(f"\n=== {coverage}x: adaptive drop = {drop_name} (idx={drop_idx}, "
              f"hits={hits[drop_idx]:,}) ===")

        windowed_npy, bins_path = seo.window_fixed_drop(
            raw_npy, outdir, drop_idx=drop_idx, max_hit_frac=None, retain_counts=False)

        pi_arr, pj_arr, het_scale_val, het_scale_diag = seo.run_inference_diploid(
            windowed_npy, device, ckpt_path=Path(P.CKPT_DIPLOID), kind="ril2")

        bed_dir = outdir / "bed"
        hae.write_imputed_bed(f"{INDIVIDUAL}_ril2_{TAG}", pi_arr, pj_arr, drop_idx,
                               gamete_names, bins_path, bed_dir, bin_size=BIN_SIZE)
        print(f"  -> wrote BED to {bed_dir}")


if __name__ == "__main__":
    main()
