#!/usr/bin/env python
"""
Reference-bias driver: for each (dataset, individual) at 0.1x, run the
`refmap` and `chain_ordinary_256` alignment arms and record per-index-sample
hit ratios. Alignment-stage only -- no CRF decode, no VCF scoring (that ~47
min/row tail is irrelevant to a reference-bias question). See
/home/zrm22/.claude/plans/dreamy-booping-sutton.md for the full design.

Reused, UNMODIFIED: heldout_assembly_eval.run_refmap (+ its CHAIN_ARMS dict),
heldout_assembly_eval.load_gamete_names, simval_paths (v2 index/manifest
paths), simval_eval_one.prep_fastq. v1->v2 index cutover done the same way
simval_eval_one.py does it: patch hae.nb.FMD/LIFT in-process, no shared file
edited.

New here: refbias_parse.py (ps4g-header-based attribution, not raw.tsv --
raw.tsv blanks the hit list for EXACT reads, see that module's docstring);
this driver's manifest-row lookup + per-(row,arm) JSON cache.

Usage:
    refbias_eval.py --coverage 0.1 --arms refmap,chain_ordinary_256
        [--dataset-id ID --individual NAME]  # single row; default = full 0.1x rung
        [--threads 20]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_paths as P  # noqa: E402
import refbias_parse as rp  # noqa: E402
from simval_eval_one import prep_fastq  # noqa: E402

# v1 -> v2 cutover, in-process only (same pattern as simval_eval_one.py).
hae.nb.FMD = P.FMD_V2
hae.nb.LIFT = P.LIFT_V2

SCRATCH_ROOT = P.GRITS_WORKDIR / "scratch/refbias"
DEFAULT_ARMS = ["refmap", "chain_ordinary_256"]


def load_manifest_rows(coverage, dataset_id=None, individual=None):
    rows = []
    with open(P.MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["coverage"] != coverage:
                continue
            if dataset_id and row["dataset_id"] != dataset_id:
                continue
            if individual and row["individual"] != individual:
                continue
            rows.append(row)
    return rows


def run_one(row, arm, threads):
    dataset_id, individual, coverage = row["dataset_id"], row["individual"], row["coverage"]
    row_key = f"{dataset_id}__{individual}__{coverage}x"
    outdir = SCRATCH_ROOT / f"{row_key}__{arm}"
    outdir.mkdir(parents=True, exist_ok=True)
    result_json = outdir / "refbias_result.json"
    if result_json.exists():
        return json.loads(result_json.read_text())

    hae.THREADS = str(threads)
    fastq = prep_fastq(row["r1_path"], row["r2_path"], outdir / "reads.fastq")

    total_input_reads = 2 * int(row["r1_bases"]) // 150  # R1+R2 concatenated, 150bp reads

    t0 = time.time()
    npy_path = hae.run_refmap(individual, fastq, outdir, arm=arm)
    align_s = time.time() - t0

    ps4g_path = outdir / "raw.ps4g"
    metrics = rp.compute_row_metrics(ps4g_path, total_input_reads)

    try:
        parents = rp.split_individual_name(individual, row["kind"])
    except ValueError as e:
        parents = None
        print(f"  WARNING: {e}", file=sys.stderr)

    result = {
        "dataset_id": dataset_id,
        "class": row["class"],
        "kind": row["kind"],
        "individual": individual,
        "coverage": coverage,
        "realized_coverage": row["realized_coverage"],
        "arm": arm,
        "parents": parents,
        "align_seconds": align_s,
        **metrics,
    }
    result_json.write_text(json.dumps(result, indent=1))
    # reads.fastq is large (100-200MB/sample) and fully reproducible from the
    # manifest paths -- delete it once its ps4g/npy output exists, so a full
    # 40-sample x 2-arm sweep doesn't leave ~15GB of duplicate uncompressed
    # fastq behind (each individual's fastq is arm-independent but this
    # driver keeps arms in separate outdirs, matching run_refmap's own
    # per-outdir contract).
    fastq.unlink(missing_ok=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="0.1")
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--individual", default=None)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--threads", type=int, default=20)
    args = ap.parse_args()

    arms = args.arms.split(",")
    rows = load_manifest_rows(args.coverage, args.dataset_id, args.individual)
    if not rows:
        raise SystemExit(f"no manifest rows matched coverage={args.coverage} "
                          f"dataset_id={args.dataset_id} individual={args.individual}")

    print(f"{len(rows)} manifest row(s) x {len(arms)} arm(s) = {len(rows) * len(arms)} runs")
    for row in rows:
        for arm in arms:
            key = f"{row['dataset_id']}/{row['individual']}/{row['coverage']}x/{arm}"
            print(f"=== {key} ===")
            t0 = time.time()
            result = run_one(row, arm, args.threads)
            print(f"    placement_rate={result['placement_rate']:.4f} "
                  f"B73_hit_ratio={result['hit_ratio'].get('B73', float('nan')):.4f} "
                  f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
