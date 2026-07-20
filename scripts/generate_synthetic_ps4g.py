#!/usr/bin/env python
"""Generate a synthetic PS4G file for benchmarking the HMM path finders.

Simulates a single haplotype "trajectory" across a contig with occasional
recombination breakpoints, then emits read-count evidence (a mix of
unambiguous single-gamete rows and ambiguous multi-gamete rows) consistent
with that trajectory at every binned position - similar in spirit to real
read-mapping evidence, without needing real reads/assemblies.

Usage:
    python scripts/generate_synthetic_ps4g.py --positions 10000 --gametes 10 \\
        --output /tmp/bench_medium.ps4g --seed 1
"""

import argparse
import random
from pathlib import Path


def generate(positions: int, gametes: int, contig: str, seed: int, recomb_prob: float = 0.001):
    rng = random.Random(seed)
    gamete_names = [f"gamete{i}:0" for i in range(gametes)]

    current_gamete = rng.randrange(gametes)
    rows = []
    gamete_totals = [0] * gametes

    for pos in range(1, positions + 1):
        if rng.random() < recomb_prob:
            current_gamete = rng.randrange(gametes)

        n_rows = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_rows):
            count = rng.randint(1, 30)
            if rng.random() < 0.75 or gametes == 1:
                # Unambiguous: supports the true current gamete.
                indices = [current_gamete]
            else:
                # Ambiguous: true gamete plus 1-2 random decoys.
                n_extra = rng.randint(1, min(2, gametes - 1))
                decoys = rng.sample([g for g in range(gametes) if g != current_gamete], n_extra)
                indices = sorted([current_gamete] + decoys)
            rows.append((indices, pos, count))
            for idx in indices:
                gamete_totals[idx] += count

    return gamete_names, gamete_totals, rows


def write_ps4g(output_path: Path, contig: str, gamete_names, gamete_totals, rows, command: str):
    total_counts = sum(count for _indices, _pos, count in rows)
    with open(output_path, "w") as fh:
        fh.write("#PS4G\n")
        fh.write("#version=2.0\n")
        fh.write(f"#Command: {command}\n")
        fh.write(f"#TotalUniqueCounts: {total_counts}\n")
        fh.write("#gamete\tgameteIndex\tcount\n")
        for idx, name in enumerate(gamete_names):
            fh.write(f"#{name}\t{idx}\t{gamete_totals[idx]}\n")
        fh.write("gameteSet\trefContig\trefPosBinned\tcount\n")
        for indices, pos, count in rows:
            fh.write(f"{','.join(map(str, indices))}\t{contig}\t{pos}\t{count}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, required=True, help="Number of binned positions to generate.")
    parser.add_argument("--gametes", type=int, default=10, help="Number of reference gametes.")
    parser.add_argument("--contig", default="chr1", help="Contig/chromosome name.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--recomb-prob", type=float, default=0.001, help="Per-position probability of a simulated recombination breakpoint.")
    parser.add_argument("--output", required=True, help="Output PS4G file path.")
    args = parser.parse_args()

    gamete_names, gamete_totals, rows = generate(args.positions, args.gametes, args.contig, args.seed, args.recomb_prob)
    command = f"generate_synthetic_ps4g.py --positions {args.positions} --gametes {args.gametes} --seed {args.seed}"
    write_ps4g(Path(args.output), args.contig, gamete_names, gamete_totals, rows, command)
    print(f"Wrote {len(rows)} data rows ({args.positions} positions, {args.gametes} gametes) to {args.output}")


if __name__ == "__main__":
    main()
