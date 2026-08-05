#!/usr/bin/env python3
"""
Sort a BEAGLE map file by chr, then pos, then cM.
Remove rows where cM decreases as pos increases (within each chr).
Split output into one file per chr.

Usage: python clean_sort_map.py <input.map> <output_dir>
"""
import sys
import os
from collections import defaultdict

def chr_sort_key(chr_name):
    """Sort chr1, chr2, ... chrN numerically."""
    num = chr_name.replace("chr", "").replace("Chr", "")
    try:
        return (0, int(num))
    except ValueError:
        return (1, num)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.map> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Read all rows
    rows_by_chr = defaultdict(list)
    with open(input_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            chr_name = parts[0]
            cm = float(parts[2])
            pos = int(parts[3])
            rows_by_chr[chr_name].append((pos, cm, parts))

    total_removed = 0

    for chr_name in sorted(rows_by_chr, key=chr_sort_key):
        # Sort by pos ascending, then cM ascending as tiebreak
        rows = sorted(rows_by_chr[chr_name], key=lambda r: (r[0], r[1]))

        # Remove rows where cM decreases relative to the last kept row
        kept = []
        max_cm = -float("inf")
        removed = 0
        for pos, cm, parts in rows:
            if cm >= max_cm:
                kept.append(parts)
                max_cm = cm
            else:
                removed += 1

        total_removed += removed

        out_path = os.path.join(output_dir, f"{chr_name}.map")
        with open(out_path, "w") as out:
            for parts in kept:
                out.write("\t".join(parts) + "\n")

        print(f"{chr_name}: {len(kept)} rows kept, {removed} removed -> {out_path}")

    print(f"\nTotal rows removed: {total_removed}")

if __name__ == "__main__":
    main()
