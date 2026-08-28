#!/usr/bin/env python
"""
Zero-sequencing-error control for all 5 IDX-RIL2 pairs at 0.5x, both
bin-size conventions (256 default / 1). For each pair: build zero-error
0.5x reads (scripts/simval_noerr_ril2_reads.py, reusing cached mosaic
segments), align with bin_size in {None, 1} via simval_eval_one.do_align
(adaptive drop self-determined, kind="ril2"), then score against the
unchanged truth mosaic (reads don't affect truth) via extract_error_
regions.extract_wrong_intervals. Writes the full wrong-interval list per
(pair, tag) plus one summary table comparing against the existing
with-sequencing-error 0.5x baselines already on disk from this session.

Run under phg-ml (needs torch for inference):
  LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
    /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/run_ril2_noerr_sweep.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402
from extract_error_regions import extract_wrong_intervals  # noqa: E402
from simval_oracle_bed import build_ril_mosaics  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT

PAIRS = ["B73xOh43", "B73xCML103", "B97xCML103", "Il14HxB97", "Oh43xIl14H"]
COVERAGE = "0.5"
NOERR_READS_ROOT = P.SCRATCH_ROOT.parent / "noerr_ril2_reads"
OUT_DIR = Path("results/ril2_error_regions/noerr_0.5x")
BIN_SIZE_TAGS = [(None, "unfiltered-bin-noerr"), (1, "binsize1-noerr")]
WITH_ERROR_TAGS = ["unfiltered-bin", "binsize1"]  # existing baselines, reused not rebuilt


def build_reads(pair):
    parent_a, parent_b = pair.split("x")
    outdir = NOERR_READS_ROOT / pair
    gz_r1 = outdir / f"{pair}.{COVERAGE}x.R1.fastq.gz"
    gz_r2 = outdir / f"{pair}.{COVERAGE}x.R2.fastq.gz"
    if not (gz_r1.exists() and gz_r2.exists()):
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "simval_noerr_ril2_reads.py"),
            "--parent-a", parent_a, "--parent-b", parent_b,
            "--coverage", COVERAGE, "--error-rate", "0", "--outdir", str(outdir),
        ], check=True)
    return str(gz_r1), str(gz_r2)


def align_one(pair, tag, bin_size, r1, r2):
    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{pair}__{COVERAGE}x__{tag}"
    args = argparse.Namespace(
        sample=f"{pair}_ril2_{tag}", r1=r1, r2=r2, outdir=str(outdir),
        arm="refmap", bin_size=bin_size, no_cleanup=False, drop_idx=None,
        max_hit_frac=None, retain_counts=False, kind="ril2", ckpt=str(P.CKPT_DIPLOID),
    )
    if (outdir / "bed").exists() and any((outdir / "bed").glob("*.bed")):
        print(f"[{pair}/{tag}] SKIP align, bed/ already populated")
        return outdir
    print(f"[{pair}/{tag}] aligning...")
    info = seo.do_align(args)
    print(f"[{pair}/{tag}] {info}")
    return outdir


def score(pair, tag, outdir):
    parent_a, parent_b = pair.split("x")
    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", parent_a, parent_b)
    bed_dir = outdir / "bed"
    sample = f"{pair}_ril2_{tag}"
    wrong, wrong_bp, total_bp = extract_wrong_intervals(bed_dir, sample, mosaic_h1, label_to_name)
    return wrong, wrong_bp, total_bp


def existing_baseline(pair, tag):
    """Reuse the already-built with-sequencing-error BED for this (pair,tag)
    if present -- no rebuild, pure re-scoring of existing output."""
    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{pair}__{COVERAGE}x__{tag}"
    if not (outdir / "bed").exists():
        return None
    wrong, wrong_bp, total_bp = score(pair, tag, outdir)
    return len(wrong), wrong_bp, total_bp


def process_pair(pair):
    """Run both bin-sizes for one pair end-to-end; write this pair's own
    partial summary file (not the merged SUMMARY.*) so concurrent worker
    subprocesses for different pairs never write-contend on the same file --
    aggregate() merges all partials in one single-threaded pass afterward."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r1, r2 = build_reads(pair)
    rows = []
    for bin_size, tag in BIN_SIZE_TAGS:
        outdir = align_one(pair, tag, bin_size, r1, r2)
        wrong, wrong_bp, total_bp = score(pair, tag, outdir)
        error_pct = wrong_bp / total_bp * 100 if total_bp else float("nan")

        region_file = OUT_DIR / f"{pair}_{tag}_error_regions.json"
        region_file.write_text(json.dumps(wrong, indent=1))
        print(f"[{pair}/{tag}] n_wrong={len(wrong)} wrong_bp={wrong_bp:,} "
              f"total_bp={total_bp:,} error_pct={error_pct:.4f}%  -> {region_file}")

        with_error_tag = "unfiltered-bin" if bin_size is None else "binsize1"
        baseline = existing_baseline(pair, with_error_tag)
        base_n_wrong, base_wrong_bp, base_total_bp = baseline if baseline else (None, None, None)
        base_pct = (base_wrong_bp / base_total_bp * 100) if baseline else None

        rows.append({
            "pair": pair, "tag": tag, "bin_size": bin_size if bin_size else 256,
            "n_wrong": len(wrong), "wrong_bp": wrong_bp, "total_bp": total_bp,
            "error_pct_noerr": round(error_pct, 4),
            "n_wrong_with_error": base_n_wrong, "wrong_bp_with_error": base_wrong_bp,
            "error_pct_with_error": round(base_pct, 4) if base_pct is not None else None,
            "with_error_tag_used": with_error_tag if baseline else "NOT AVAILABLE",
        })
    (OUT_DIR / f"_partial_{pair}.json").write_text(json.dumps(rows, indent=1))
    return rows


def aggregate():
    summary = []
    for pair in PAIRS:
        partial = OUT_DIR / f"_partial_{pair}.json"
        if not partial.exists():
            print(f"WARNING: no partial summary for {pair}, skipping in aggregate")
            continue
        summary.extend(json.loads(partial.read_text()))

    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    with open(OUT_DIR / "SUMMARY.tsv", "w") as f:
        cols = list(summary[0].keys())
        f.write("\t".join(cols) + "\n")
        for row in summary:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")

    lines = ["# Zero-sequencing-error RIL2 sweep, 0.5x, all 5 pairs x 2 bin-sizes", "",
              "| pair | bin_size | n_wrong | error % (no-error) | error % (with error) |",
              "|---|---|---|---|---|"]
    for row in summary:
        we = f"{row['error_pct_with_error']:.4f}%" if row["error_pct_with_error"] is not None else "n/a"
        lines.append(f"| {row['pair']} | {row['bin_size']} | {row['n_wrong']} | "
                      f"{row['error_pct_noerr']:.4f}% | {we} |")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_DIR}/SUMMARY.{{json,tsv,md}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None, help="run only this one pair (worker mode); omit to run all + aggregate")
    ap.add_argument("--aggregate-only", action="store_true", help="skip processing, just merge existing partials")
    cli = ap.parse_args()

    if cli.aggregate_only:
        aggregate()
    elif cli.pair:
        process_pair(cli.pair)
    else:
        for pair in PAIRS:
            process_pair(pair)
        aggregate()


if __name__ == "__main__":
    main()
