from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
import random
import pandas as pd
import pysam

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
def pick_crossovers(length: int,
                    min_spacing: int = 1_000_000,
                    max_spacing: int = 9_000_000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Vectorized: draw inter-event distances ~ Uniform[min,max], cumsum, trim.
       Guarantees last tail <= max by inserting extra points if needed."""
    rng = np.random.default_rng() if rng is None else rng
    if not (0 < min_spacing < max_spacing <= length):
        return np.array([], dtype=np.int64)

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
    return pos

# ----------------------------
# Mosaic utilities
# ----------------------------
def make_founder_genome(chrom_lengths: Dict[Union[int, str], int], founder_id: str) -> Line:
    """Each chromosome starts as a single interval from this founder."""
    return {c: [Interval(0, L, founder_id)] for c, L in chrom_lengths.items()}

def _intervals_covering(mosaic: Mosaic, s: int, e: int) -> List[Interval]:
    """Return the intervals that overlap with [s,e). Mosaics are continuous; s,e within bounds."""
    # Binary search could be added; linear is fine if K is modest.
    intervals = []
    for iv in mosaic:
        if iv.end > s and iv.start < e:
            intervals.append(iv)

    if len(intervals) > 0: return intervals
    else:
        print("s: ", s)
        print("e: ", e)
        print("mosaic: ", mosaic)
        print("intervals: ", intervals)
        raise RuntimeError("Requested child interval not covered by parent mosaic.")

def recombine_two_mosaics(mA: Mosaic, mB: Mosaic,
                          length: int,
                          cross_idx: np.ndarray,
                          start_with_A: bool = None,
                          rng: np.random.Generator | None = None) -> (Mosaic, Mosaic):
    """Build child mosaics from parents A and B, alternating at edges."""
    rng = np.random.default_rng() if rng is None else rng
    if start_with_A is None:
        start_with_A = bool(rng.integers(0, 2))
    edges = np.concatenate(([0], cross_idx, [length]))
    starts, ends = edges[:-1], edges[1:]

    child1: Mosaic = []
    child2: Mosaic = []
    use_A = start_with_A

    for s, e in zip(starts, ends):
        if not s < e: raise RuntimeError("end idx before start idx")

        donor1 = mA if use_A else mB
        donor2 = mB if use_A else mA

        ivs1= _intervals_covering(donor1, int(s), int(e))
        ivs2= _intervals_covering(donor2, int(s), int(e))

        # Keep ancestry label from donor interval (founder id)
        for iv1 in ivs1:
            if iv1.start <= s < e <= iv1.end:
                child1.append(Interval(int(s), int(e), iv1.founder))
            elif s <= iv1.start < e <= iv1.end:
                child1.append(Interval(int(iv1.start), int(e), iv1.founder))
            elif s <= iv1.start < iv1.end <= e:
                child1.append(Interval(int(iv1.start), int(iv1.end), iv1.founder))
            else:
                child1.append(Interval(int(s), int(iv1.end), iv1.founder))

        for iv2 in ivs2:
            if iv2.start <= s < e <= iv2.end:
                child2.append(Interval(int(s), int(e), iv2.founder))
            elif s <= iv2.start < e <= iv2.end:
                child2.append(Interval(int(iv2.start), int(e), iv2.founder))
            elif s <= iv2.start < iv2.end <= e:
                child2.append(Interval(int(iv2.start), int(iv2.end), iv2.founder))
            else:
                child2.append(Interval(int(s), int(iv2.end), iv2.founder))

        use_A = not use_A

    return child1, child2

def cross_lines(lineA: Line, lineB: Line,
                chrom_lengths: Dict[Union[int, str], int],
                rng: np.random.Generator | None = None) -> (Line, Line):
    """One haploid child genome from two haploid parental lines."""
    rng = np.random.default_rng() if rng is None else rng
    child1: Line = {}
    child2: Line = {}
    for c, L in chrom_lengths.items():
        cross_idx = pick_crossovers(L, 1_000_000, 9_000_000, rng=rng)  # 5Mb-ish spacing
        child1[c], child2[c] = recombine_two_mosaics(lineA[c], lineB[c], L, cross_idx, start_with_A=None, rng=rng)
    return child1, child2

# ----------------------------
# Simulation loop
# ----------------------------
def simulate_rounds(chrom_lengths: Dict[Union[int, str], int],
                    founders: List[str],
                    rounds: int,
                    rng_seed: int | None = None) -> List[Line]:
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
    pop: List[Line] = [make_founder_genome(chrom_lengths, f) for f in founders]

    for r in range(rounds):
        random.shuffle(pop)  # in-place pairing
        next_pop: List[Line] = []
        # Pair adjacent lines and produce two children to keep size constant
        for i in range(0, len(pop), 2):
            A, B = pop[i], pop[i+1]
            child1, child2 = cross_lines(A, B, chrom_lengths, rng=rng)
            next_pop.extend([child1, child2])
        pop = next_pop
    return pop

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
def create_df(line: Line) -> pd.DataFrame:
    starts = []
    ends = []
    parents = []
    chromosome = []

    for c, mosaic in line.items():
        for interval in mosaic:
            # Assemble DataFrame
            starts.append(interval.start)
            ends.append(interval.end)
            parents.append(interval.founder)
            chromosome.append(c)

    df = pd.DataFrame({
        "chr": chromosome,
        "start": starts,
        "end": ends,
        "parent": parents
    })

    return df

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

def shift_chrom_arm(pop, c, length):
    for i, m in enumerate(pop):
        for interval in m[c]:
            interval.start += length
            interval.end += length

def merge_pop(pop1, pop2):
    pop = []
    for i, m in enumerate(pop1):
        line = {}
        for c in m:
            line[c]= m[c] + pop2[i][c]
        pop.append(line)
    return pop

def count_gaps(seq, bases):
    gaps = 0
    counter = 0
    for base in seq:
        if counter > bases: break
        if base == "-": gaps += 1
        else: counter += 1
    return gaps

def convert_coord(s_info, ref_info, ref_start, ref_end):
    """
    s_info["start"] is the parent alignment start
    ref_info["start"] is the ref alignment start
    ref_start is the desired ref start
    s_info["end"] is the parent alignment end
    ref_info["end"] is the ref alignment end
    ref_end is the desired ref end
    """

    # CASE 1: desired sequence starts before s and ends before s
    if ref_start < ref_info["start"] and ref_end <= ref_info["end"]:
        bases_to_ref_end = ref_end - ref_info["start"]
        gaps_to_ref_end = count_gaps(ref_info["seq"], bases_to_ref_end)  # number of gaps in ref before reaching desired end
        parent_idx_end = bases_to_ref_end + gaps_to_ref_end
        gaps_to_parent_end = count_gaps(s_info["seq"][:parent_idx_end], len(s_info["seq"][:parent_idx_end]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"]
            parent_end = s_info["start"] + parent_idx_end - gaps_to_parent_end
        else:
            parent_start = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_end = parent_start + parent_idx_end - gaps_to_parent_end

    # CASE 2: desired sequence starts before s and ends after s
    elif ref_start < ref_info["start"] and ref_end > ref_info["end"]:
        if s_info["strand"] == "+":
            parent_start = s_info["start"]
            parent_end = s_info["end"]
        else:
            parent_start = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_end = parent_start + s_info["length"]

    # CASE 3: desired sequence starts after s and ends before s (fully contained)
    elif ref_start >= ref_info["start"] and ref_end <= ref_info["end"]:
        bases_to_ref_start = ref_start - ref_info["start"]
        gaps_to_ref_start = count_gaps(ref_info["seq"], bases_to_ref_start)  # number of gaps in ref before reaching desired start
        parent_idx_start = bases_to_ref_start + gaps_to_ref_start
        gaps_to_parent_start = count_gaps(s_info["seq"][:parent_idx_start], len(s_info["seq"][:parent_idx_start]))

        bases_to_ref_end = ref_end - ref_info["start"]
        gaps_to_ref_end = count_gaps(ref_info["seq"], bases_to_ref_end)  # number of gaps in ref before reaching desired end
        parent_idx_end = bases_to_ref_end + gaps_to_ref_end
        gaps_to_parent_end = count_gaps(s_info["seq"][:parent_idx_end], len(s_info["seq"][:parent_idx_end]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"] + parent_idx_start - gaps_to_parent_start
            parent_end = s_info["start"] + parent_idx_end - gaps_to_parent_end
        else:
            parent_start_seq = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_start = parent_start_seq + parent_idx_start - gaps_to_parent_start
            parent_end = parent_start_seq + parent_idx_end - gaps_to_parent_end

    # CASE 4: desired sequence starts after s and ends after s
    elif ref_start >= ref_info["start"] and ref_end > ref_info["end"]:
        bases_to_ref_start = ref_start - ref_info["start"]
        gaps_to_ref_start = count_gaps(ref_info["seq"], bases_to_ref_start)  # number of gaps in ref before reaching desired start
        parent_idx_start = bases_to_ref_start + gaps_to_ref_start
        gaps_to_parent_start = count_gaps(s_info["seq"][:parent_idx_start], len(s_info["seq"][:parent_idx_start]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"] + parent_idx_start - gaps_to_parent_start
            parent_end = s_info["end"]
        else:
            parent_start_seq = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_start = parent_start_seq + parent_idx_start - gaps_to_parent_start
            parent_end = parent_start_seq + s_info["length"]

    else:
        parent_start = None
        parent_end = None

    if parent_start == parent_end:
        parent_start = None
        parent_end = None

    return parent_start, parent_end

def parse_maf(maf_file, ref_chr, ref_start, ref_end):
    ref = False
    ref_info = None # (chrom, start, length, end, strand, chr_len, seq)

    parent_coordinates = []

    with open(maf_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("a"):
                continue

            elif line.startswith("s"): # first is ref, second is parent
                ref = not ref # is this the ref or the parent?
                #s_fields = line.strip().split("\t")
                s_fields = [f.strip() for f in line.strip().split("\t")]
                s_info = {
                    "chr": s_fields[1],
                    "start": int(s_fields[2]),
                    "end": int(s_fields[2]) + int(s_fields[3]), # noninclusive end
                    "length": int(s_fields[3]),
                    "strand": s_fields[4],
                    "chr_length": int(s_fields[5]),
                    "seq": s_fields[6]
                }

                if ref:
                    ref_fields = line.strip().split("\t")
                    if str(ref_fields[1]) == str(ref_chr) and (int(ref_fields[2]) + int(ref_fields[3]) >= ref_start and int(ref_fields[2]) <= ref_end):

                        ref_info = {
                            "chr": ref_fields[1],
                            "start": int(ref_fields[2]),
                            "end": int(ref_fields[2]) + int(ref_fields[3]),
                            "length": int(ref_fields[3]),
                            "strand": ref_fields[4],
                            "chr_length": int(ref_fields[5]),
                            "seq": ref_fields[6]
                        }

                    else: ref_info = None

                elif ref_info is not None:
                    parent_chr = s_info["chr"]
                    parent_start, parent_end = convert_coord(s_info, ref_info, ref_start, ref_end)
                    parent_coordinates.append((parent_chr,parent_start,parent_end))
                    ref_info = None

    return parent_coordinates

def maf_table(maf_file):
    cols = [
        "ref_chr", "ref_start", "ref_end", "ref_length", "ref_strand", "ref_chr_length", "ref_seq",
        "s_chr",   "s_start",   "s_end",   "s_length",   "s_strand",   "s_chr_length",   "s_seq"
    ]

    rows = []
    ref_info = None  # holds the previous 's' line (reference)

    with open(maf_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "a")):
                # skip comments and alignment headers; only care about 's' lines
                continue

            if line.startswith("s"):
                # Split on ANY whitespace to prevent trailing spaces from sneaking in
                fields = [f.strip() for f in line.strip().split("\t")]
                # MAF 's' format: s src start size strand srcSize text
                entry = {
                    "chr":        fields[1],
                    "start":      int(fields[2]),
                    "length":     int(fields[3]),
                    "strand":     fields[4],
                    "chr_length": int(fields[5]),
                    "seq":        fields[6],
                }
                entry["end"] = entry["start"] + entry["length"]  # non-inclusive end

                if ref_info is None:
                    # First 's' in a pair -> reference
                    ref_info = entry
                else:
                    # Second 's' in a pair -> sample; emit a row, then reset
                    s_info = entry
                    rows.append({
                        "ref_chr":        ref_info["chr"],
                        "ref_start":      ref_info["start"],
                        "ref_end":        ref_info["end"],
                        "ref_length":     ref_info["length"],
                        "ref_strand":     ref_info["strand"],
                        "ref_chr_length": ref_info["chr_length"],
                        "ref_seq":        ref_info["seq"],
                        "s_chr":          s_info["chr"],
                        "s_start":        s_info["start"],
                        "s_end":          s_info["end"],
                        "s_length":       s_info["length"],
                        "s_strand":       s_info["strand"],
                        "s_chr_length":   s_info["chr_length"],
                        "s_seq":          s_info["seq"],
                    })
                    ref_info = None  # ready for next pair

    return pd.DataFrame(rows, columns=cols)

def parse_maf_table(maf_table, ref_chr, ref_start, ref_end):
    parent_coordinates = []
    maf_table_subset = maf_table[
        (maf_table["ref_chr"] == ref_chr) &
        (maf_table["ref_end"] >= ref_start) &
        (maf_table["ref_start"] <= ref_end)
        ]

    for index, row in maf_table_subset.iterrows():
        s_info = {
            "chr": row["s_chr"],
            "start": row["s_start"],
            "end": row["s_end"],
            "length": row["s_length"],
            "strand": row["s_strand"],
            "chr_length": row["s_chr_length"],
            "seq": row["s_seq"]
        }
        ref_info = {
            "chr": row["ref_chr"],
            "start": row["ref_start"],
            "end": row["ref_end"],
            "length": row["ref_length"],
            "strand": row["ref_strand"],
            "chr_length": row["ref_chr_length"],
            "seq": row["ref_seq"]
        }
        parent_chr = row["s_chr"]
        parent_start, parent_end = convert_coord(s_info, ref_info, ref_start, ref_end)
        if parent_start is not None:
            parent_coordinates.append((parent_chr,parent_start,parent_end))
    return parent_coordinates

def adjust_coords(df, length):
    df.loc[df.index[0], "parent_start"] = 0
    end = df.loc[df.index[0], "parent_end"]

    # iterate from second row onward
    for i in df.index[1:]:
        if df.at[i, "parent_start"] != end:
            df.at[i, "parent_start"] = end
        end = df.at[i, "parent_end"]

    # set last end to full chromosome length
    df.loc[df.index[-1], "parent_end"] = length

    return df


def old_convert_ref_coords(parent, parent_chroms):
    maf_file = f"/workdir/smm477/uncrossed_phg/alignment_files/{parent}.maf"
    fasta_key = f"{parent}_fasta_key.bed"
    parent_df = pd.read_csv(f"{parent}_key.bed", sep="\t", header=None, names=["chr", "start", "end", "founder"])

    for i, row in parent_df.iterrows():
        parent_coords = parse_maf(maf_file, row["chr"], row["start"], row["end"])
        for c, start, end in parent_coords:
            coord_data = {'chr': [c], 'start': [start], 'end': [end], 'founder': [row['founder']]}
            coord_df = pd.DataFrame(coord_data)
            coord_df.to_csv(fasta_key, sep="\t", index=False, mode='a', header=False)

    unsorted_df = pd.read_csv(fasta_key, sep="\t", header=None, names=["chr", "start", "end", "founder"])
    sorted_df = unsorted_df.sort_values(by=["parent_chr", "parent_start", "parent_end"], ascending=[True, True, True])

    for c, length in parent_chroms.items():
        adjusted_df = adjust_coords(sorted_df[sorted_df["chr"] == c], length)
        adjusted_df.to_csv(f"{parent}_fasta_key_adjusted.bed", sep="\t", index=False, header=False, mode="a")


def convert_ref_coords(parent, parent_chroms):
    maf_file = f"{parent}.maf"
    #maf_file = f"/workdir/smm477/uncrossed_phg/alignment_files/{parent}.maf"
    parent_df = pd.read_csv(f"{parent}_refkey.bed", sep="\t", header=None, names=["chr", "start", "end", "founder"])
    # want to preserve the original index, df is sorted by ref_chr, ref_start, ref_end

    key_df = pd.DataFrame(columns=["ref_chr", "ref_start", "ref_end", "parent_chr", "parent_start", "parent_end", "founder"])

    for i, row in parent_df.iterrows():
        parent_coords = parse_maf(maf_file, row["chr"], row["start"], row["end"])
        for c, start, end in parent_coords:
            coord_data = {"ref_chr": [row["chr"]],
                          "ref_start": [row["start"]],
                          "ref_end": [row["end"]],
                          "parent_chr": [c],
                          "parent_start": [start],
                          "parent_end": [end],
                          "founder": [row["founder"]]}
            coord_df = pd.DataFrame(coord_data)
            key_df = pd.concat([key_df, coord_df], ignore_index=True)

    sorted_df = key_df.sort_values(by=["parent_chr", "parent_start", "parent_end"], ascending=[True, True, True])

    for c, length in parent_chroms.items():
        adjusted_df = adjust_coords(sorted_df[sorted_df["parent_chr"] == c], length)
        adjusted_df = adjusted_df[adjusted_df["parent_start"] != adjusted_df["parent_end"]] # do I want to do this or not?
        adjusted_df.to_csv(f"{parent}_key.bed", sep="\t", index=False, header=False, mode="a")

from time import perf_counter

def convert_ref_coords_fast(parent, parent_chroms):
    maf_file = f"{parent}.maf"
    parent_df = pd.read_csv(
        f"{parent}_refkey.bed",
        sep="\t", header=None,
        names=["chr", "start", "end", "founder"]
    )

    t0 = perf_counter()

    rows = []
    maf = maf_table(maf_file)

    # itertuples is faster than iterrows
    for row in parent_df.itertuples(index=False, name="RefRow"):
        # row.chr, row.start, row.end, row.founder
        #parent_coords = parse_maf(maf_file, row.chr, int(row.start), int(row.end))
        parent_coords = parse_maf_table(maf, row.chr, row.start, row.end)
        if not parent_coords:
            continue
        # parent_coords expected iterable of (c, start, end)
        else:
            for c, start, end in parent_coords:
                if end <= start:                # drop zero/negative length
                    continue
                rows.append({
                    "ref_chr": row.chr,
                    "ref_start": int(row.start),
                    "ref_end": int(row.end),
                    "parent_chr": c,
                    "parent_start": int(start),
                    "parent_end": int(end),
                    "founder": row.founder,
                })

    if not rows:
        # nothing to write; create empty file or return early
        open(f"{parent}_key.bed", "w").close()
        return

    key_df = pd.DataFrame.from_records(rows)

    # sort once
    sorted_df = key_df.sort_values(
        by=["parent_chr", "parent_start", "parent_end"],
        ascending=[True, True, True]
    )

    # adjust per chromosome, collect parts
    adjusted_parts = []
    for c, length in parent_chroms.items():
        chunk = sorted_df[sorted_df["parent_chr"] == c]
        if chunk.empty:
            continue
        adj = adjust_coords(chunk, int(length))
        # keep the zero-length guard (belt & suspenders)
        adj = adj[adj["parent_start"] != adj["parent_end"]]
        adjusted_parts.append(adj)

    if adjusted_parts:
        out_df = pd.concat(adjusted_parts, ignore_index=True)
    else:
        out_df = sorted_df

    # single write (no mode="a")
    out_df.to_csv(f"{parent}_key.bed", sep="\t", index=False, header=False)

    t1 = perf_counter()
    print(f"[convert_ref_coords] wrote {len(out_df):,} rows for {parent} in {t1 - t0:.2f}s")


def convert_ref_coords_map(parent, parent_chroms):
    maf_file = f"{parent}.maf"
    parent_df = pd.read_csv(
        f"{parent}_refkey.bed",
        sep="\t", header=None,
        names=["chr", "start", "end", "founder"]
    )

    t0 = perf_counter()

    maf = maf_table(maf_file)  # build once

    # map: per row -> list of (parent_chr, start, end)
    mapped = parent_df.copy()
    mapped["parent_coords"] = mapped.apply(
        lambda r: parse_maf_table(maf, r["chr"], r["start"], r["end"]),
        axis=1
    )

    # normalize empty results to []
    mapped["parent_coords"] = mapped["parent_coords"].apply(lambda x: x if x else [])

    # explode the lists into rows
    mapped = mapped.explode("parent_coords", ignore_index=True)

    # drop rows that had no coords after explode (NaN)
    mapped = mapped[mapped["parent_coords"].notna()]
    if mapped.empty:
        open(f"{parent}_key.bed", "w").close()
        print(f"[convert_ref_coords_map] wrote 0 rows for {parent} in {perf_counter() - t0:.2f}s")
        return

    # split tuple into columns
    mapped[["parent_chr", "parent_start", "parent_end"]] = pd.DataFrame(
        mapped["parent_coords"].tolist(), index=mapped.index
    )

    # filter zero/negative
    mapped = mapped[mapped["parent_end"] > mapped["parent_start"]]

    # choose output columns + rename to match your previous function
    key_df = mapped[[
        "chr", "start", "end", "parent_chr", "parent_start", "parent_end", "founder"
    ]].rename(columns={
        "chr": "ref_chr",
        "start": "ref_start",
        "end": "ref_end",
    })

    if key_df.empty:
        open(f"{parent}_key.bed", "w").close()
        print(f"[convert_ref_coords_map] wrote 0 rows for {parent} in {perf_counter() - t0:.2f}s")
        return

    # single global sort
    sorted_df = key_df.sort_values(
        by=["parent_chr", "parent_start", "parent_end"],
        ascending=[True, True, True]
    )

    # per-chrom adjust
    adjusted_parts = []
    for c, length in parent_chroms.items():
        chunk = sorted_df[sorted_df["parent_chr"] == c]
        if chunk.empty:
            continue
        adj = adjust_coords(chunk, int(length))
        adj = adj[adj["parent_start"] != adj["parent_end"]]
        adjusted_parts.append(adj)

    out_df = pd.concat(adjusted_parts, ignore_index=True) if adjusted_parts else sorted_df

    out_df.to_csv(f"{parent}_key.bed", sep="\t", index=False, header=False)

    t1 = perf_counter()
    print(f"[convert_ref_coords_map] wrote {len(out_df):,} rows for {parent} in {t1 - t0:.2f}s")


def write_fastas(num_founders, chromosomes):
    for i in range(num_founders):
        fasta_file = f"{i}.fa"  # create a new fasta file made up of parent fasta sequence
        with open(fasta_file, "w") as f:
            fasta_key = pd.read_csv(f"{i}_key.bed", sep="\t", header=None, names=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent"], index_col=False)
            for c in chromosomes:
                f.write(f">{c}\n")
            # print by chromosome
                for r, row in fasta_key[fasta_key["fa_chr"].astype(str) == str(c)].iterrows():
                    with pysam.FastaFile(f"{row['parent']}.fa") as fa:
                    #with pysam.FastaFile(f"/workdir/smm477/uncrossed_phg/uplated_fastas/{row["parent"]}.fa") as fa:
                        seq = fa.fetch(str(row["parent_chr"]), int(row["parent_start"]), int(row["parent_end"]))
                    f.write(f"{seq}\n")


def build_fasta_keys(parents, founder):
    fasta_df = pd.DataFrame(
        columns=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent", "seg_length"])

    founder_df = pd.DataFrame(columns=["ref_chr", "ref_start", "ref_end", "parent_chr", "parent_start", "parent_end", "founder"])
    key = []

    for parent in parents: # for each parent, subset founder and aggregate
        parent_df = pd.read_csv(f"{parent}_key.bed", sep="\t", header=None,
                                names=["ref_chr", "ref_start", "ref_end", "parent_chr", "parent_start", "parent_end", "founder"])
        parent_subset = parent_df[parent_df["founder"] == int(founder)]
        founder_df = pd.concat([founder_df, parent_subset], ignore_index=True)
        key.extend([parent]*len(parent_subset))

    founder_df["parent"] = key
    # sort based on ref_chr, ref_start, ref_end
    founder_df = founder_df.sort_values(by=["ref_chr", "ref_start", "ref_end"], ascending=[True, True, True])
    fasta_df["fa_chr"] = founder_df["ref_chr"]
    fasta_df["seg_len"] = (founder_df["parent_end"] - founder_df["parent_start"]).astype(int)
    fasta_df["fa_end"] = fasta_df.groupby("fa_chr")["seg_len"].cumsum()
    fasta_df["fa_start"] = fasta_df["fa_end"] - fasta_df["seg_len"]
    fasta_df = fasta_df.drop(columns=["seg_len"])
    fasta_df["parent_chr"] = founder_df["parent_chr"]
    fasta_df["parent_start"] = founder_df["parent_start"]
    fasta_df["parent_end"] = founder_df["parent_end"]
    fasta_df["parent"] = founder_df["parent"]

    fasta_df.to_csv(f"{founder}_key.bed", sep="\t", index=False, header=False)

# 1/2 chromosome lengths (bp)
chrom_lengths = {"chr1" : 300, "chr2" : 150, "scaf_1" : 50}

# B73_chrom_lengths = {"chr1" : 308_452_471, "chr2" : 243_675_191, "chr3" : 238_017_767, "chr4" : 250_330_460, "chr5" : 226_353_449,
#                      "chr6" : 181_357_234, "chr7" : 185_808_916, "chr8" : 182_411_202, "chr9" : 163_004_744, "chr10" : 152_435_371}
#
# B73_arm_lengths = {"chr1" : 308_452_471//2, "chr2" : 243_675_191//2, "chr3" : 238_017_767//2, "chr4" : 250_330_460//2, "chr5" : 226_353_449//2,
#                    "chr6" : 181_357_234//2, "chr7" : 185_808_916//2, "chr8" : 182_411_202//2, "chr9" : 163_004_744//2, "chr10" : 152_435_371//2}
#
# founder_chroms = {"CML228" : {"chr1" : 311_577_201, "chr2" : 244_763_794, "chr3" : 239_761_349, "chr4" : 254_676_231, "chr5" : 228_823_570,
#                               "chr6" : 175_338_271, "chr7" : 181_540_992, "chr8" : 186_718_620, "chr9" : 167_687_098, "chr10" : 149_925_833},
#                   "CML322" : {"chr1" : 304_784_548, "chr2" : 243_324_309, "chr3" : 239_728_129, "chr4" : 257_539_888, "chr5" : 221_631_323,
#                               "chr6" : 175_362_398, "chr7" : 178_688_969, "chr8" : 180_302_813, "chr9" : 164_706_825, "chr10" : 150_084_177},
#                   "CML69" : {"chr1" : 305_788_947, "chr2" : 242_308_534, "chr3" : 239_259_890, "chr4" : 255_185_478, "chr5" : 218_448_888,
#                              "chr6" : 173_207_479, "chr7" : 180_887_457, "chr8" : 180_190_510, "chr9" : 162_111_082, "chr10" : 153_170_938},
#                   "Ki11" : {"chr1" : 310_658_259, "chr2" : 252_411_040, "chr3" : 240_499_622, "chr4" : 252_694_885, "chr5" : 224_468_191,
#                             "chr6" : 178_360_981, "chr7" : 184_781_161, "chr8" : 182_190_944, "chr9" : 165_949_266, "chr10" : 153_305_303},
#                   "M162W" : {"chr1" : 306_203_930, "chr2" : 242_470_251, "chr3" : 237_156_436, "chr4" : 252_621_820, "chr5" : 222_291_023,
#                              "chr6" : 186_544_510, "chr7" : 183_763_309, "chr8" : 209_474_767, "chr9" : 166_176_350, "chr10" : 153_192_159},
#                   "Ms71" : {"chr1" : 310_202_964, "chr2" : 243_466_302, "chr3" : 240_604_834, "chr4" : 253_912_153, "chr5" : 224_736_366,
#                             "chr6" : 179_969_880, "chr7" : 184_671_976, "chr8" : 179_905_066, "chr9" : 163_431_469, "chr10" : 149_277_564},
#                   "Oh43" : {"chr1" : 306_440_704, "chr2" : 248_330_814, "chr3" : 240_829_167, "chr4" : 253_228_108, "chr5" : 221_656_378,
#                             "chr6" : 179_113_859, "chr7" : 180_662_582, "chr8" : 181_014_384, "chr9" : 167_198_378, "chr10" : 152_214_488},
#                   "B97" : {"chr1" : 307_632_032, "chr2" : 252_179_072, "chr3" : 242_347_007, "chr4" : 252_232_367, "chr5" : 222_239_690,
#                            "chr6" : 180_431_575, "chr7" : 182_842_045, "chr8" : 183_646_229, "chr9" : 165_161_870, "chr10" : 150_691_437},
#                   "CML247" : {"chr1" : 328_910_756, "chr2" : 263_791_054, "chr3" : 240_585_047, "chr4" : 254_335_221, "chr5" : 225_397_941,
#                               "chr6" : 178_806_624, "chr7" : 181_641_468, "chr8" : 185_239_947, "chr9" : 164_3086_29, "chr10" : 153_508_852},
#                   "CML333" : {"chr1" : 314_488_983, "chr2" : 249_665_033, "chr3" : 243_215_679, "chr4" : 251_519_228, "chr5" : 225_739_712,
#                               "chr6" : 180_183_266, "chr7" : 184_022_950, "chr8" : 187_782_559, "chr9" : 173_472_473, "chr10" : 151_644_700},
#                   "HP301" : {"chr1" : 307_047_731, "chr2" : 250_141_472, "chr3" : 240_039_523, "chr4" : 252_291_229, "chr5" : 221_343_133,
#                              "chr6" : 177_507_800, "chr7" : 181_725_512, "chr8" : 178_051_958, "chr9" : 164_342_879, "chr10" : 151_729_224},
#                   "Ki3" : {"chr1" : 312_622_006, "chr2" : 246_815_332, "chr3" : 245_550_371, "chr4" : 252_658_346, "chr5" : 227_524_873,
#                            "chr6" : 188_595_388, "chr7" : 179_755_421, "chr8" : 180_35_2562, "chr9" : 164_864_428, "chr10" : 152_190_761},
#                   "M37W" : {"chr1" : 306_885_972, "chr2" : 244_314_634, "chr3" : 242_405_911, "chr4" : 251_302_864, "chr5" : 221_795_845,
#                             "chr6" : 186_724_520, "chr7" : 183_487_981, "chr8" : 180_780_255, "chr9" : 166_919_533, "chr10" : 155_388_826},
#                   "NC350" : {"chr1" : 310_842_591, "chr2" : 249_001_196, "chr3" : 243_851_070, "chr4" : 254_309_806, "chr5" : 223_249_382,
#                              "chr6" : 182_993_951, "chr7" : 180_105_666, "chr8" : 181_843_295, "chr9" : 169_120_828, "chr10" : 151_937_703},
#                   "Oh7B" : {"chr1" : 307_239_180, "chr2" : 245_704_635, "chr3" : 237_596_221, "chr4" : 251_422_589, "chr5" : 223_847_857,
#                             "chr6" : 180_001_022, "chr7" : 184_573_146, "chr8" : 182_109_197, "chr9" : 205_794_433, "chr10" : 111_214_652},
#                   "Tzi8" : {"chr1" : 305_803_203, "chr2" : 245_299_735, "chr3" : 239_545_720, "chr4" : 254_136_052, "chr5" : 221_425_360,
#                             "chr6" : 178_662_525, "chr7" : 184_517_021, "chr8" : 182_126_902, "chr9" : 167_326_923, "chr10" : 154_632_701},
#                   "CML103" : {"chr1" : 305_897_857, "chr2" : 241_481_138, "chr3" : 238_563_298, "chr4" : 252_598_624, "chr5" : 222_463_916,
#                               "chr6" : 176_230_381, "chr7" : 181_096_865, "chr8" : 179_3524_51, "chr9" : 166_148_394, "chr10" : 149_914_000},
#                   "CML277" : {"chr1" : 308_867_798, "chr2" : 247_283_677, "chr3" : 242_596_678, "chr4" : 250_135_365, "chr5" : 220_513_029,
#                               "chr6" : 176_052_463, "chr7" : 181_030_607, "chr8" : 182_825_486, "chr9" : 167_429_005, "chr10" : 150_946_910},
#                   "CML52" : {"chr1" : 318_156_096, "chr2" : 259_708_713, "chr3" : 243_974_402, "chr4" : 262_669_390, "chr5" : 231707303,
#                              "chr6" : 182_704_949, "chr7" : 186_094_419, "chr8" : 190_099_871, "chr9" : 169_172_938, "chr10" : 153_584_098},
#                   "Il14H" : {"chr1" : 300_139_798, "chr2" : 244_722_702, "chr3" : 235_706_095, "chr4" : 257_978_302, "chr5" : 223_951_407,
#                              "chr6" : 171_793_723, "chr7" : 178_410_452, "chr8" : 178_543_749, "chr9" : 167_710_400, "chr10" : 152_911_007},
#                   "Ky21" : {"chr1" : 311_254_152, "chr2" : 243_376_689, "chr3" : 237_194_928, "chr4" : 247_764_743, "chr5" : 219_724_524,
#                             "chr6" : 174_022_551, "chr7" : 182_197_498, "chr8" : 178_924_872, "chr9" : 166_334_496, "chr10" : 150_109_315},
#                   "Mo18W" : {"chr1" : 307_773_640, "chr2" : 247_669_541, "chr3" : 244_467_206, "chr4" : 252_758_625, "chr5" : 226_800_839,
#                              "chr6" : 180_377_583, "chr7" : 180_330_399, "chr8" : 182_323_521, "chr9" : 164_132_304, "chr10" : 148_387_378},
#                   "NC358" : {"chr1" : 306_333_774, "chr2" : 277_154_671, "chr3" : 239_682_009, "chr4" : 253_506_590, "chr5" : 221_841_946,
#                              "chr6" : 184_556_742, "chr7" : 181_107_323, "chr8" : 177_955_777, "chr9" : 165_009_261, "chr10" : 156_023_696},
#                   "P39" : {"chr1" : 302_421_781, "chr2" : 244_619_812, "chr3" : 242_478_718, "chr4" : 275_636_967, "chr5" : 222_867_812,
#                            "chr6" : 177_971_375, "chr7" : 206_991_990, "chr8" : 176_984_287, "chr9" : 164_153_970, "chr10" : 148_196_188}}

# 2N founders: e.g., 100 founders (2N = 100). Provide any labels you like.
founders = ["B97", "CML69", "Tzi8", "Mo37W"]  # 2N founders

# NAM_founders = ["CML228", "CML322", "CML69", "Ki11", "M162W", "Ms71", "Oh43", "B97", "CML247", "CML333", "HP301", "Ki3",
#                 "M37W", "NC350", "Oh7B", "Tzi8", "CML103", "CML277", "CML52", "Il14H", "Ky21", "Mo18W", "NC358", "P39"]
#
# Run ~1250 rounds to simulate land race (cross ~4000 bp)
#landrace_pop = simulate_rounds(B73_arm_lengths, NAM_founders, rounds=1250)
landrace_pop = simulate_rounds(chrom_lengths, founders, rounds=1000)
for line in landrace_pop:
    print(line)


# Run once to get two parent cross (cross ~ 5 Mbp)
#two_parent_pop = simulate_rounds(B73_arm_lengths, NAM_founders, rounds=1)
two_parent_pop = simulate_rounds(chrom_lengths, founders, rounds=1)

# For each chromosome, shift either landrace or two_parent by chrom_length
# for chrom, length in B73_arm_lengths.items():
#     shift = random.choice([0, 1]) # randomly choose 0 or 1
#     if shift: shift_chrom_arm(landrace_pop, chrom, length)
#     else: shift_chrom_arm(two_parent_pop, chrom, length)
#

# for chrom, length in chrom_lengths.items():
#     shift = random.choice([0, 1]) # randomly choose 0 or 1
#     if shift: shift_chrom_arm(landrace_pop, chrom, length)
#     else: shift_chrom_arm(two_parent_pop, chrom, length)
#
# pop = merge_pop(landrace_pop, two_parent_pop)
# convert_pop_to_key(pop, founders)
#
# for founder in founders:
#     #convert_ref_coords(founder, {"chr1" : 650, "chr2" : 280, "scaf_1" : 105})
#     convert_ref_coords_map(founder, {"chr1" : 650, "chr2" : 280, "scaf_1" : 105})
#
#
# for i in range(len(founders)):
#     build_fasta_keys(founders, i)
#
# write_fastas(len(founders), chrom_lengths.keys())






#
# convert_pop_to_key(pop, NAM_founders)
#
# for founder in NAM_founders:
#     convert_ref_coords(founder, founder_chroms[founder])

# print(maf_table("B97.maf"))