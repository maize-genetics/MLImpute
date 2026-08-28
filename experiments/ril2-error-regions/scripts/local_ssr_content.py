#!/usr/bin/env python
"""
Local SSR (simple sequence repeat / tandem repeat) content around each
error region, via Tandem Repeats Finder (trf, installed at
/programs/bin/util/trf -- no existing SSR/repeat annotation exists for
this reference). Extracts a window (default +-5000bp) around each error
interval's midpoint from B73.fa via samtools faidx, runs trf in -ngs
(multi-sequence, compact stdout) mode with standard parameters
(2 7 7 80 10 50 500), and reports each window's total bp covered by
tandem repeats and repeat count.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

FASTA = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa"
FAI = FASTA + ".fai"


def load_chrom_lengths():
    lengths = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            lengths[name] = int(length)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors-json", required=True)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--out-json", required=True)
    cli = ap.parse_args()

    errors = json.loads(Path(cli.errors_json).read_text())
    lengths = load_chrom_lengths()

    regions = []
    for i, err in enumerate(errors):
        mid = (err["start"] + err["end"]) // 2
        s = max(0, mid - cli.window)
        e = min(lengths[err["chrom"]], mid + cli.window)
        region = f"{err['chrom']}:{s + 1}-{e}"  # samtools faidx is 1-based inclusive
        regions.append(region)

    regions_file = "/tmp/ssr_regions.txt" if False else Path(cli.out_json).with_suffix(".regions.txt")
    regions_file.write_text("\n".join(regions) + "\n")

    fasta_out = Path(cli.out_json).with_suffix(".windows.fa")
    with open(fasta_out, "w") as out_f:
        subprocess.run(["samtools", "faidx", "-r", str(regions_file), FASTA],
                        stdout=out_f, check=True)

    trf_out = subprocess.run(
        ["/programs/bin/util/trf", str(fasta_out), "2", "7", "7", "80", "10", "50", "500",
         "-d", "-h", "-ngs"],
        capture_output=True, text=True)
    # -ngs returns 0 on success per its own help text; still check output non-empty
    lines = trf_out.stdout.splitlines()

    # -ngs format: a line "@<seqname>" starts each sequence's block, followed by
    # 0+ repeat lines: start end period copies consensusSize percentMatches ...
    per_region = {r: {"n_repeats": 0, "repeat_bp": 0} for r in regions}
    current = None
    for line in lines:
        if line.startswith("@"):
            current = line[1:].strip()
            if current not in per_region:
                per_region[current] = {"n_repeats": 0, "repeat_bp": 0}
        elif line.strip() and current is not None:
            parts = line.split()
            try:
                start, end = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
            per_region[current]["n_repeats"] += 1
            per_region[current]["repeat_bp"] += (end - start + 1)

    out = []
    for i, err in enumerate(errors):
        region = regions[i]
        s, e = region.split(":")[1].split("-")
        window_bp = int(e) - int(s) + 1
        stats = per_region.get(region, {"n_repeats": 0, "repeat_bp": 0})
        out.append({
            **err, "ssr_window": region,
            "ssr_n_repeats": stats["n_repeats"],
            "ssr_repeat_bp": stats["repeat_bp"],
            "ssr_fraction": round(stats["repeat_bp"] / window_bp, 4),
        })

    Path(cli.out_json).write_text(json.dumps(out, indent=1))
    n_with_ssr = sum(1 for o in out if o["ssr_n_repeats"] > 0)
    print(f"processed {len(out)} regions; {n_with_ssr} have >=1 tandem repeat in window")
    print(f"wrote {cli.out_json}")


if __name__ == "__main__":
    main()
