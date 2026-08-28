#!/usr/bin/env python
"""
Simulate one labeled hybrid read set (Oh43xIl14H, IDX-HYB, 0.1x-equivalent)
at a caller-chosen wgsim error rate, for the zero-sequencing-error control
described in /home/zrm22/.claude/plans/deep-jumping-salamander.md.

Not a corpus-driver replacement: writes into a private scratch dir, never
touches the shared corpus tree or manifest.tsv (build_read_datasets.py's
manifest_has() would silently skip re-appending over a deleted/rebuilt row --
see the plan's "Files" section). Reuses the corpus's own simlib primitives
unmodified (simulate_reads_pe already exposes error_rate= as a kwarg).

0.1x is normally a 5% nested subsample of a 2.0x master (config.py's
TOP_COVERAGE/nested_subsample_and_gzip) -- reproducing that exactly would mean
re-simulating 14.34M read pairs to keep 717K, AND changing -e desyncs wgsim's
RNG stream at the first error event regardless, so a read-for-read match with
the baseline is unattainable either way. Simulates 0.05x directly per parent
instead (matches the corpus's per-parent coverage-balanced hybrid mix,
build_read_datasets.build_hybrid_master's convention, just skipping the
2.0x-master + subsample detour) -- 716,836 pairs total vs baseline's 716,952
(-0.016%, verified negligible).

Usage:
    simval_noerr_reads.py --error-rate 0 --seed-tag master_half \
        --outdir scratch/noerr_reads/arm_b_noerror
    simval_noerr_reads.py --error-rate 0.001 --seed-tag master_half_redraw \
        --outdir scratch/noerr_reads/arm_c_redraw
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import simval_truth_labels as tl  # noqa: E402

CORPUS_SCRIPTS = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts"
)
sys.path.insert(0, str(CORPUS_SCRIPTS))
import config as cfg  # noqa: E402
import simlib  # noqa: E402

DATASET_ID = "IDX-HYB"
PARENT_A = "Oh43"
PARENT_B = "Il14H"
INDIVIDUAL = f"{PARENT_A}x{PARENT_B}"
TARGET_COVERAGE = 0.05  # per parent -> 0.1x combined hybrid, matches build_hybrid_master's
                        # per-parent-coverage convention (half of TOP_COVERAGE there == 1.0x
                        # per parent for the 2.0x master; here we go straight to the 0.1x point)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--error-rate", required=True,
                     help="wgsim -e; corpus default is 0.001, pass 0 for the no-error arm")
    ap.add_argument("--seed-tag", default="master_half",
                     help="simlib.seed_for(dataset_id, sample, <tag>) -- use the corpus's own "
                          "'master_half' to match its exact seeding convention, or a distinct "
                          "tag for an independent redraw control")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--no-label", action="store_true",
                     help="skip L1 read renaming (plain concat) -- default is labeled")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(exist_ok=True)

    parts_r1, parts_r2, per_parent = [], [], {}
    for sample in (PARENT_A, PARENT_B):
        fasta = cfg.assembly_path(sample, False)
        fai = fasta.with_suffix(fasta.suffix + ".fai")
        genome_bp = simlib.assembly_size(fai)
        n_pairs = simlib.coverage_to_pairs(TARGET_COVERAGE, genome_bp)
        seed = simlib.seed_for(DATASET_ID, sample, args.seed_tag)

        r1 = raw_dir / f"{sample}.R1.fastq"
        r2 = raw_dir / f"{sample}.R2.fastq"
        simlib.simulate_reads_pe(fasta, n_pairs, seed, r1, r2, error_rate=str(args.error_rate))

        parts_r1.append((r1, sample))
        parts_r2.append((r2, sample))
        per_parent[sample] = {"genome_bp": genome_bp, "n_pairs": n_pairs, "seed": seed}
        print(f"[{sample}] {n_pairs} pairs, seed={seed}, genome_bp={genome_bp}", flush=True)

    r1_out = outdir / f"{INDIVIDUAL}.0.1x.R1.fastq.gz"
    r2_out = outdir / f"{INDIVIDUAL}.0.1x.R2.fastq.gz"

    if args.no_label:
        simlib.concat_files([p for p, _ in parts_r1], r1_out.with_suffix(""))
        simlib.concat_files([p for p, _ in parts_r2], r2_out.with_suffix(""))
        import subprocess
        subprocess.run(["gzip", "-f", str(r1_out.with_suffix(""))], check=True)
        subprocess.run(["gzip", "-f", str(r2_out.with_suffix(""))], check=True)
        r1_counts = r2_counts = {s: per_parent[s]["n_pairs"] for s in per_parent}
    else:
        _, r1_counts = tl.rename_and_concat(parts_r1, r1_out)
        _, r2_counts = tl.rename_and_concat(parts_r2, r2_out)

    manifest = {
        "dataset_id": DATASET_ID, "individual": INDIVIDUAL, "coverage": "0.1",
        "error_rate": args.error_rate, "seed_tag": args.seed_tag,
        "labeled": not args.no_label,
        "per_parent": per_parent,
        "r1_path": str(r1_out), "r2_path": str(r2_out),
        "r1_read_counts": r1_counts, "r2_read_counts": r2_counts,
        "r1_bases": sum(r1_counts.values()) * cfg.READ_LEN,
        "r2_bases": sum(r2_counts.values()) * cfg.READ_LEN,
        "genome_bp_mean": (per_parent[PARENT_A]["genome_bp"] + per_parent[PARENT_B]["genome_bp"]) / 2,
    }
    manifest["realized_coverage"] = (
        (manifest["r1_bases"] + manifest["r2_bases"]) / manifest["genome_bp_mean"]
    )
    (outdir / "sim_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
