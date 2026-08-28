#!/usr/bin/env python
"""
Sample N random genomic regions (chrom-length-weighted, same interval-width
distribution as a given error list) to serve as a background/control set --
raw "194/210 windows contain an SSR" is meaningless without knowing the
baseline rate; same for N-gap distance and PAV-proxy overlap rates.
"""
import argparse
import json
import random
from pathlib import Path

FAI = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai"


def load_chrom_lengths():
    lengths = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            if name.startswith("chr") and name[3:].isdigit():
                lengths[name] = int(length)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors-json", required=True, help="reference error list, for width distribution")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    cli = ap.parse_args()

    errors = json.loads(Path(cli.errors_json).read_text())
    widths = [max(1, e["width"]) for e in errors]
    lengths = load_chrom_lengths()
    chroms = list(lengths.keys())
    total_len = sum(lengths.values())
    weights = [lengths[c] / total_len for c in chroms]

    rng = random.Random(cli.seed)
    out = []
    for i in range(cli.n_samples):
        chrom = rng.choices(chroms, weights=weights, k=1)[0]
        width = rng.choice(widths)
        s = rng.randint(0, max(0, lengths[chrom] - width))
        out.append({"chrom": chrom, "start": s, "end": s + width, "width": width,
                     "decoded_parent1": None, "decoded_parent2": None, "true_founder": None})

    Path(cli.out_json).write_text(json.dumps(out, indent=1))
    print(f"wrote {cli.n_samples} background regions to {cli.out_json}")


if __name__ == "__main__":
    main()
