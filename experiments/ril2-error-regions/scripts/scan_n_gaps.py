#!/usr/bin/env python
"""
Scan B73.fa (chr1-chr10) for N-runs (assembly gaps) using the .fai offsets
for direct seeking -- no existing N-gap annotation exists for this
reference (confirmed by search), so this builds one from scratch. Writes a
BED of [chrom, start, end] for every maximal run of N/n bases.
"""
import argparse
import re
import sys
from pathlib import Path

FASTA = Path("/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa")
FAI = FASTA.with_suffix(".fa.fai")
CHROMS = [f"chr{i}" for i in range(1, 11)]


def load_fai(fai_path):
    entries = {}
    with open(fai_path) as f:
        for line in f:
            name, length, offset, linebases, linewidth = line.split("\t")
            entries[name] = (int(length), int(offset), int(linebases), int(linewidth))
    return entries


def read_sequence(fasta_path, name, entry):
    length, offset, linebases, linewidth = entry
    n_lines = -(-length // linebases)  # ceil
    n_bytes = n_lines * linewidth
    with open(fasta_path, "rb") as f:
        f.seek(offset)
        raw = f.read(n_bytes)
    seq = raw.replace(b"\n", b"").decode("ascii")[:length]
    assert len(seq) == length, f"{name}: expected {length} got {len(seq)}"
    return seq


def find_n_runs(seq):
    return [(m.start(), m.end()) for m in re.finditer(r"[Nn]+", seq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    fai = load_fai(FAI)
    with open(cli.out, "w") as out:
        for chrom in CHROMS:
            seq = read_sequence(FASTA, chrom, fai[chrom])
            runs = find_n_runs(seq)
            for s, e in runs:
                out.write(f"{chrom}\t{s}\t{e}\n")
            print(f"{chrom}: {len(runs)} N-runs, total {sum(e - s for s, e in runs):,} bp", file=sys.stderr)

    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
