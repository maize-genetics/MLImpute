#!/usr/bin/env python
"""
Generic zero-(or arbitrary-)sequencing-error IDX-RIL2 read builder, for an
arbitrary (parent_a, parent_b, coverage) -- generalizes the IDX-HYB-only,
whole-genome `simval_noerr_reads.py` precedent to RIL2's mosaic-segment
simulation (`build_read_datasets.build_ril2_master`'s code path), which
has no equivalent in that script (no mosaic logic there at all).

Reuses each pair's already-extracted, cached mosaic-segment FASTA at
scratch/read_datasets/IDX-RIL2/{pair}/{A,B}_{founder}.segments.fa (present
on disk for all 5 official pairs -- these already encode the EXACT mosaic/
breakpoints/founder-assignment from the official build, since extraction
only depends on (dataset_id, parent_a, parent_b, replicate) via
`mosaic.derive_ril2_mosaic`'s deterministic seeding, not on coverage or
error rate) -- so no mosaic re-derivation, liftover, or Projector needed
here at all, just simlib's read-simulation primitives.

Mirrors `build_ril2_master`'s own per-founder loop exactly (same seed_for
key, same concat-not-relabel convention -- RIL2's official reads carry NO
founder-label read-name prefix, unlike IDX-HYB's `rename_and_concat`
scheme, confirmed by inspecting real raw.tsv read names this session):
  seed = simlib.seed_for(dataset_id, "A"|"B", founder_name, "ril2_reads", replicate)
  simlib.simulate_reads_pe(segments_fa, n_pairs, seed, r1, r2, error_rate=...)
  simlib.concat_files([...], master.R{1,2}.fastq)

Only `n_pairs` (via `--coverage`, direct simulation rather than the
official "simulate 2.0x then nested-subsample" path) and `error_rate` can
differ from the official build here -- everything else is bit-identical.
Writes ONLY under a local --outdir (never the shared corpus tree or its
manifest.tsv).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts")
import simlib  # noqa: E402

READ_DATASETS_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/read_datasets/IDX-RIL2")


def gzip_file(src, dst):
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as out_f:
        proc = subprocess.run(["pigz", "-c", "-p", "8", str(src)], stdout=out_f, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"pigz failed for {src}: {proc.stderr[-2000:].decode(errors='replace')}")
    tmp.rename(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="IDX-RIL2")
    ap.add_argument("--parent-a", required=True)
    ap.add_argument("--parent-b", required=True)
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--error-rate", default="0")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    cli = ap.parse_args()

    pair = f"{cli.parent_a}x{cli.parent_b}"
    src_dir = READ_DATASETS_ROOT / pair
    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gz_r1 = outdir / f"{pair}.{cli.coverage}x.R1.fastq.gz"
    gz_r2 = outdir / f"{pair}.{cli.coverage}x.R2.fastq.gz"
    if gz_r1.exists() and gz_r2.exists():
        print(f"[{pair}] already built: {gz_r1}")
        return str(gz_r1), str(gz_r2)

    part_r1, part_r2 = [], []
    for label, sample in (("A", cli.parent_a), ("B", cli.parent_b)):
        segments_fa = src_dir / f"{label}_{sample}.segments.fa"
        if not segments_fa.exists():
            raise FileNotFoundError(f"missing cached segments FASTA: {segments_fa}")
        seg_bp = simlib.fasta_total_bp(segments_fa)
        n_pairs = simlib.coverage_to_pairs(cli.coverage, seg_bp)
        seed = simlib.seed_for(cli.dataset_id, label, sample, "ril2_reads", cli.replicate)
        r1 = outdir / f"{label}_{sample}.R1.fastq"
        r2 = outdir / f"{label}_{sample}.R2.fastq"
        print(f"[{pair}] {label}={sample}: seg_bp={seg_bp:,} n_pairs={n_pairs:,} "
              f"seed={seed} error_rate={cli.error_rate}")
        simlib.simulate_reads_pe(segments_fa, n_pairs, seed, r1, r2, error_rate=cli.error_rate)
        part_r1.append(r1)
        part_r2.append(r2)

    master_r1 = outdir / "master.R1.fastq"
    master_r2 = outdir / "master.R2.fastq"
    simlib.concat_files(part_r1, master_r1)
    simlib.concat_files(part_r2, master_r2)

    gzip_file(master_r1, gz_r1)
    gzip_file(master_r2, gz_r2)
    for f in (master_r1, master_r2, *part_r1, *part_r2):
        Path(f).unlink(missing_ok=True)

    manifest = {
        "dataset_id": cli.dataset_id, "individual": pair, "coverage": cli.coverage,
        "error_rate": cli.error_rate, "replicate": cli.replicate,
        "r1_path": str(gz_r1), "r2_path": str(gz_r2),
    }
    (outdir / "sim_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {gz_r1}\nwrote {gz_r2}")
    return str(gz_r1), str(gz_r2)


if __name__ == "__main__":
    main()
