#!/usr/bin/env python3
"""
Usage: python fix_bed_boundaries.py <chr_lengths_file> <bed_file> > output.bed

Sets the start of the first entry for each chromosome to 0,
and the end of the last entry to the chromosome length.
"""

import sys
from collections import defaultdict

chr_lengths_file = sys.argv[1]
bed_file = sys.argv[2]

# --- Step 1: Load chromosome lengths ---
# File format: ">chr1 308452471"
chr_lengths = {}
with open(chr_lengths_file) as f:
    for line in f:
        parts = line.strip().split()
        chrom = parts[0].lstrip(">")   # remove leading ">"
        length = int(parts[1])
        chr_lengths[chrom] = length

# --- Step 2: Load BED file, grouping rows by chromosome ---
# Each row is stored as a list of fields so we can modify individual values
header = None
rows_by_chrom = defaultdict(list)
chrom_order = []   # track the order chromosomes appear

with open(bed_file) as f:
    for line in f:
        fields = line.strip().split("\t")
        if fields[0] == "chrom":
            header = fields
            continue
        chrom = fields[0]
        if chrom not in rows_by_chrom:
            chrom_order.append(chrom)
        rows_by_chrom[chrom].append(fields)

# --- Step 3: Fix boundaries and print ---
print("\t".join(header))

for chrom in chrom_order:
    rows = rows_by_chrom[chrom]

    rows[0][1] = "0"                              # first row: set start to 0
    rows[-1][2] = str(chr_lengths[chrom])         # last row: set end to chr length

    for row in rows:
        print("\t".join(row))
