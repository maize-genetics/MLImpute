#!/usr/bin/env python
"""
Whole-chromosome-mosaic RIL sanity check (see
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md).

Builds the simplest possible RIL-like individual for Oh43xIl14H: each of
chr1..chr10 is assigned WHOLE to one parent, alternating (chr1=Oh43,
chr2=Il14H, chr3=Oh43, ...), and BOTH haplotypes get the identical
assignment (homozygous per chromosome -- no recombination, no
heterozygosity, no breakpoint-position edge cases at all). This isolates
one question: does the model correctly and confidently call the right
founder for an entire, unambiguous chromosome?

Reuses, UNMODIFIED: mosaic.Projector/load_b73_chrom_lengths, simlib's
extraction/wgsim/concat helpers (the same primitives
build_read_datasets.build_ril_master uses -- just fed a hand-built
deterministic mosaic instead of draw_breakpoints/build_haplotype_mosaic's
random draws), and grits_workdir/scripts/simval_eval_one.py's --stage align
(align+window+infer+BED-write in one call, no --stage score / gVCF
comparison at all, per the plan).

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/build_chr_mosaic_ril.py
"""
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

CORPUS_SCRIPTS = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts")
sys.path.insert(0, str(CORPUS_SCRIPTS))
import config as cfg  # noqa: E402
import mosaic  # noqa: E402
import simlib  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SCRATCH = GRITS_WORKDIR / "scratch/chr_mosaic_ril"
OUTDIR = GRITS_WORKDIR / "scratch/simval_eval/CHRMOS-RIL__Oh43xIl14H__0.1x"
DATASET_TAG = "CHRMOS-RIL"
SAMPLE = "Oh43xIl14H_chrmos"
PARENT_A = ("Oh43", False)   # (sample, is_heldout) -- both IDX_SAMPLES
PARENT_B = ("Il14H", False)
TARGET_COVERAGE = 0.1
PYTHON = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
TEST_CRF_SRC = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src"


def build_deterministic_mosaic(chrom_lengths):
    """chr1..chr10 in cfg.B73_CHROMS order, alternating whole-chromosome
    founder starting at A=Oh43. Both haplotypes get the SAME assignment
    (homozygous per chromosome)."""
    mosaic_h = {}
    expected = []
    for i, chrom in enumerate(cfg.B73_CHROMS):
        founder = "A" if i % 2 == 0 else "B"
        mosaic_h[chrom] = [(0, chrom_lengths[chrom], founder)]
        expected.append((chrom, founder))
    return mosaic_h, expected


def build_reads(mosaic_h1, mosaic_h2, work_dir):
    label_to_parent = {"A": PARENT_A, "B": PARENT_B}
    projectors = {}
    for sample, is_heldout in (PARENT_A, PARENT_B):
        projectors[sample] = (None if sample == "B73"
                               else mosaic.Projector(cfg.anchorspro_path(sample, is_heldout)))

    part_r1, part_r2 = [], []
    for hap_label, hap_mosaic in (("h1", mosaic_h1), ("h2", mosaic_h2)):
        for founder in ("A", "B"):
            sample, is_heldout = label_to_parent[founder]
            proj = projectors[sample]
            b73_segs = mosaic.segments_by_founder(hap_mosaic, founder)
            query_bed = []
            for chrom, start, end in b73_segs:
                if proj is None:
                    query_bed.append((chrom, start, end))
                else:
                    query_bed.extend(proj.project_segment(chrom, start, end))
            if not query_bed:
                continue
            query_bed = simlib.merge_regions(query_bed)
            fasta = cfg.assembly_path(sample, is_heldout)
            extracted = work_dir / f"{hap_label}_{founder}_{sample}.segments.fa"
            simlib.extract_fasta_regions(fasta, query_bed, extracted)
            seg_bp = simlib.fasta_total_bp(extracted)
            n_pairs = simlib.coverage_to_pairs(TARGET_COVERAGE / 2, seg_bp)
            seed = simlib.seed_for(DATASET_TAG, hap_label, founder, sample)
            r1 = work_dir / f"{hap_label}_{founder}_{sample}.R1.fastq"
            r2 = work_dir / f"{hap_label}_{founder}_{sample}.R2.fastq"
            print(f"  {hap_label}/{founder} ({sample}): {seg_bp:,} bp extracted, "
                  f"{n_pairs:,} read pairs @ seed={seed}")
            simlib.simulate_reads_pe(extracted, n_pairs, seed, r1, r2)
            part_r1.append(r1)
            part_r2.append(r2)

    combined_r1 = work_dir / "master.R1.fastq"
    combined_r2 = work_dir / "master.R2.fastq"
    simlib.concat_files(part_r1, combined_r1)
    simlib.concat_files(part_r2, combined_r2)
    return combined_r1, combined_r2


def gzip_file(src, dst):
    dst = Path(dst)
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, "rb") as f_in, gzip.open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp.rename(dst)
    return dst


def run_align(r1_gz, r2_gz):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTDIR / "align_result.json"
    cmd = [
        PYTHON, str(GRITS_WORKDIR / "scripts/simval_eval_one.py"),
        "--stage", "align",
        "--sample", SAMPLE,
        "--r1", str(r1_gz), "--r2", str(r2_gz),
        "--outdir", str(OUTDIR),
        "--out-json", str(out_json),
        "--kind", "ril",
        "--drop-idx", "23",
    ]
    print("Running:", " ".join(cmd))
    env = {"LD_LIBRARY_PATH": "", "PYTHONPATH": TEST_CRF_SRC, "PATH": "/usr/bin:/bin:/usr/local/bin"}
    import os
    env = {**os.environ, **env}
    subprocess.run(cmd, check=True, env=env)
    return out_json


def check_results(expected):
    import pandas as pd
    expected_map = dict(expected)
    print("\n=== Expected assignment ===")
    for chrom, founder in expected:
        name = PARENT_A[0] if founder == "A" else PARENT_B[0]
        print(f"  {chrom}: {name}")

    print("\n=== Imputed BED check ===")
    all_pass = True
    for chrom, founder in expected:
        expected_name = PARENT_A[0] if founder == "A" else PARENT_B[0]
        bed_path = OUTDIR / "bed" / f"{SAMPLE}_{chrom}_imputed.bed"
        if not bed_path.exists():
            print(f"  {chrom}: MISSING {bed_path}")
            all_pass = False
            continue
        df = pd.read_csv(bed_path, sep="\t")
        n_rows = len(df)
        dominant = df.assign(span=df["end"] - df["start"]).groupby(
            ["parent1", "parent2"])["span"].sum().idxmax()
        p1, p2 = dominant
        ok = (p1 == expected_name and p2 == expected_name)
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {chrom}: expected={expected_name}  n_intervals={n_rows}  "
              f"dominant=({p1},{p2})  [{status}]")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    chrom_lengths = mosaic.load_b73_chrom_lengths()
    mosaic_h1, expected = build_deterministic_mosaic(chrom_lengths)
    mosaic_h2 = mosaic_h1  # identical -> homozygous per chromosome

    print("=== Deterministic whole-chromosome mosaic (H1 == H2) ===")
    for chrom, founder in expected:
        name = PARENT_A[0] if founder == "A" else PARENT_B[0]
        print(f"  {chrom}: {name} ({chrom_lengths[chrom]:,} bp)")

    r1, r2 = build_reads(mosaic_h1, mosaic_h2, SCRATCH)
    r1_gz = gzip_file(r1, SCRATCH / "master.R1.fastq.gz")
    r2_gz = gzip_file(r2, SCRATCH / "master.R2.fastq.gz")
    print(f"gzipped reads: {r1_gz} {r2_gz}")

    run_align(r1_gz, r2_gz)
    check_results(expected)


if __name__ == "__main__":
    main()
