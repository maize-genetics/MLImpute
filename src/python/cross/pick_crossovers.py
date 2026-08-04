from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
import random
import pandas as pd
import argparse
from cross.chrom_lengths import chrom_lengths, chrom_lengths_dicts

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Interval:
    start: int
    end: int
    founder: str  # original founder ID

Mosaic = List[Interval]                 # intervals sorted, non-overlapping
Genome = Dict[Union[int, str], Mosaic]  # chrom -> mosaic
Line   = Genome                         # alias: a "line" is a genome of mosaics

# ----------------------------
# Crossover generation (fast, bounded, tail-safe)
# ----------------------------
def pick_crossovers(A: str, B: str, length: int,
                    min_spacing: int = 1_000_000,
                    max_spacing: int = 9_000_000,
                    rng: np.random.Generator | None = None) -> List:
    """Vectorized: draw inter-event distances ~ Uniform[min,max], cumsum, trim.
       Guarantees last tail <= max by inserting extra points if needed."""
    rng = np.random.default_rng() if rng is None else rng
    if not (0 < min_spacing < max_spacing <= length):
        return []

    mean_step = 0.5 * (min_spacing + max_spacing)
    est = int(length / mean_step) + 4
    first_step = rng.integers(0, max_spacing + 1, size=1)
    steps = rng.integers(min_spacing, max_spacing + 1, size=est)
    steps = np.concatenate((first_step, steps))
    pos = np.cumsum(steps)
    pos = pos[pos < length].astype(np.int64)

    # Ensure final tail in [min, max] by inserting near the end if needed.
    last = 0 if pos.size == 0 else int(pos[-1])
    tail = length - last
    while tail > max_spacing:
        # place another crossover so that remaining tail stays >= min_spacing
        step_upper = min(max_spacing, tail - min_spacing)
        step = int(rng.integers(min_spacing, step_upper + 1))
        last = last + step
        pos = np.append(pos, last)
        tail = length - last
    return [(idx, A, B) for idx in pos]

# ----------------------------
# Simulation loop
# ----------------------------
def simulate_rounds(chrom_lengths: Dict[Union[int, str], int],
                    founders: List[str],
                    rounds: int,
                    min_spacing: int = 1_000_000,
                    max_spacing: int = 9_000_000,
                    rng_seed: int | None = None) -> Dict:
    """
    Start with 2N founders (one line per founder), then:
      - group into N pairs and cross -> 2N crossed lines
      - regroup into N pairs and cross again
      - repeat for `rounds` rounds.
    Track ancestry in interval founder labels.
    Returns the final population (size = 2N).
    """
    rng = np.random.default_rng(rng_seed)

    # Initialize 2N lines (one per founder id), each with single-interval mosaics

    pop = dict([(f, {}) for f in founders])

    for c, L in chrom_lengths.items():
        all_crosses = []
        for r in range(rounds):
            random.shuffle(founders)
            for i in range(0, len(founders), 2):
                A, B = founders[i], founders[i+1]
                all_crosses.extend(pick_crossovers(A, B, L, min_spacing=min_spacing, max_spacing=max_spacing, rng=rng))
        # crossovers must be sorted in ascending order
        all_crosses.sort(key=lambda pos: pos[0])

        lines = dict([(f, [Interval(0, L, f)]) for f in founders])

        # we build the lines
        for crossover in all_crosses:
            pos = crossover[0]
            lineA = lines[crossover[1]]
            lineB = lines[crossover[2]]

            if lineA[-1].start >= pos or lineB[-1].start >= pos:
                # in order to prevent zero-length intervals, skip crossover where the start lines up exactly with the crossover
                continue

            newA = Interval(pos, L, lineA[-1].founder)
            newB = Interval(pos, L, lineB[-1].founder)

            lineA[-1].end = pos
            lineB[-1].end = pos

            lineB.append(newA)
            lineA.append(newB)

        for f in founders:
            pop[f][c] = lines[f]

    return list(pop.values())

# ----------------------------
# Diagnostics / summaries
# ----------------------------
def mean_segment_size(line: Line) -> float:
    total_len = 0
    total_segs = 0
    for c, mosaic in line.items():
        for iv in mosaic:
            total_len += (iv.end - iv.start)
            total_segs += 1
    return total_len / max(total_segs, 1)

def founder_contributions(line: Line) -> Dict[str, int]:
    """Count segments per founder id in this line."""
    from collections import Counter
    cnt = Counter()
    for mosaic in line.values():
        for iv in mosaic:
            cnt[iv.founder] += (iv.end - iv.start)
    return dict(cnt)

# ----------------------------
# File creation
# ----------------------------

"""
convert the population from intervals to keyfiles with ref coordinates
"""
def convert_pop_to_key(pop, parents):
    # create bed keyfiles for each parent
    for i, m in enumerate(pop):
        for c in m:
            for interval in m[c]:
                # save interval to founder keyfile
                interval_data = {'chr': [c], 'start': [interval.start], 'end': [interval.end], 'founder': [i]}
                interval_df = pd.DataFrame(interval_data)
                interval_df.to_csv(f"{interval.founder}_refkey.bed", sep="\t", index=False, mode='a', header=False)

    # sort the keyfiles
    for parent in parents:
        unsorted_df = pd.read_csv(f"{parent}_refkey.bed", sep="\t", header=None, names=["chr", "start", "end", "founder"])
        (unsorted_df.sort_values(by=["chr", "start"], ascending=[True, True]).to_csv(f"{parent}_refkey.bed", sep="\t", index=False, header=False))

"""
shift the coordinates by length (move these chromosome coordinates to the lower chromsome arm)
"""
def shift_chrom_arm(pop, c, length):
    for i, m in enumerate(pop):
        for interval in m[c]:
            interval.start += length
            interval.end += length

"""
merge the two populations into one population
for each chromosome, either pop1 or pop2 has been shifted
"""
def merge_pop(pop1, pop2):
    pop = []
    for i, m in enumerate(pop1):
        line = {}
        for c in m:
            line[c]= m[c] + pop2[i][c]
        pop.append(line)
    return pop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-fasta", type=str, help="full file path to reference fasta")
    parser.add_argument("--assembly-list", type=str, help="file containing full file paths and names for assembly fastas")
    parser.add_argument("--min-spacing", type=int, default=1_000_000, help="minimum spacing between crossovers")
    parser.add_argument("--max-spacing", type=int, default=9_000_000, help="maximum spacing between crossovers")
    parser.add_argument("--rounds", type=int, nargs='+', default=[1, 10, 50, 125],
                        help="number of crossover rounds for chromosome segment, number of segments will be length of list")
    args = parser.parse_args()

    assembly_founder_paths = []
    assembly_founders = []

    with open(args.assembly_list) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                assembly_founder_paths.append(parts[0])
                assembly_founders.append(parts[1])

    ref_chrom_lengths = chrom_lengths(args.ref_fasta, exclude_scaffolds=True)

    # segment chromosome lengths (bp) as "arm" lengths
    ref_arm_lengths = {chrom: length // len(args.rounds) for chrom, length in ref_chrom_lengths.items()}

    pops = []
    for round in args.rounds:
        pop = simulate_rounds(ref_arm_lengths, assembly_founders, rounds=round, min_spacing=args.min_spacing, max_spacing=args.max_spacing)
        pops.append(pop)

    if not pops: # Handle edge case of no rounds
        merged_pop = None
    else:
        # For each chromosome, assign each pop to one region of the chromosome
        shift = np.arange(len(pops))
        for chrom, length in ref_arm_lengths.items():
            random.shuffle(shift)
            for i in range(len(pops)):
                shift_chrom_arm(pops[i], chrom, length * shift[i])

        merged_pop = pops[0]
        for pop in pops[1:]:
            merged_pop = merge_pop(pop, merged_pop)

    if merged_pop is not None:
        convert_pop_to_key(merged_pop, assembly_founders)