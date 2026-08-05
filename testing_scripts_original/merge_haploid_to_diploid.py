############
# DONT USE #
############


#!/usr/bin/env python3
"""
Merge paired haploid VCF columns (hap1/hap2) into a single diploid column.

Sample names are expected to follow the pattern: {base}.hap1.{suffix} / {base}.hap2.{suffix}
The diploid genotype is formed as: hap2_GT/hap1_GT (preserving column order).
Only the GT field is merged; for other FORMAT fields the hap1 value is used.
"""

import sys
import gzip
import argparse
import re
from collections import defaultdict


def open_vcf(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def base_name(sample):
    """Strip .hapN. or .hapN suffix to get the base sample name."""
    return re.sub(r"\.hap\d+(\..+)?$", "", sample)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Input VCF (plain or .gz); use - for stdin")
    p.add_argument("-o", "--output", default="-", help="Output VCF (default: stdout)")
    return p.parse_args()


def main():
    args = parse_args()

    in_fh = sys.stdin if args.input == "-" else open_vcf(args.input)
    out_fh = sys.stdout if args.output == "-" else open(args.output, "w")

    samples = []
    # Maps base_name -> {hap_number -> column_index}
    hap_map = {}
    fixed_cols = 9  # #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT

    for line in in_fh:
        if line.startswith("##"):
            out_fh.write(line)
            continue

        if line.startswith("#CHROM"):
            fields = line.rstrip("\n").split("\t")
            samples = fields[fixed_cols:]

            # Build hap_map
            for i, s in enumerate(samples):
                m = re.search(r"\.hap(\d+)", s)
                if m:
                    base = base_name(s)
                    hap_n = int(m.group(1))
                    if base not in hap_map:
                        hap_map[base] = {}
                    hap_map[base][hap_n] = i

            # Determine output sample order (paired bases only, sorted)
            paired = {b: h for b, h in hap_map.items() if len(h) >= 2}
            unpaired = [s for i, s in enumerate(samples)
                        if base_name(s) not in paired]

            if not paired:
                sys.exit("ERROR: No hap1/hap2 pairs found in sample names.")

            out_samples = list(paired.keys()) + unpaired
            out_fh.write("\t".join(fields[:fixed_cols] + out_samples) + "\n")
            continue

        # Data lines
        fields = line.rstrip("\n").split("\t")
        fixed = fields[:fixed_cols]
        gts = fields[fixed_cols:]

        fmt = fixed[8].split(":")
        gt_idx = fmt.index("GT") if "GT" in fmt else 0

        new_gts = []
        # Write paired diploid columns
        for base in sorted(hap_map):
            if len(hap_map[base]) < 2:
                continue
            # Use hap2 as first allele, hap1 as second (matches column order in example)
            idx2 = hap_map[base].get(2)
            idx1 = hap_map[base].get(1)
            g2 = gts[idx2].split(":")[gt_idx] if idx2 is not None else "."
            g1 = gts[idx1].split(":")[gt_idx] if idx1 is not None else "."
            diploid_gt = f"{g2}/{g1}"

            # For other FORMAT fields, take from hap1
            if len(fmt) > 1 and idx1 is not None:
                other = gts[idx1].split(":")
                other[gt_idx] = diploid_gt
                new_gts.append(":".join(other))
            else:
                new_gts.append(diploid_gt)

        # Pass through any unpaired samples unchanged
        paired_bases = set(hap_map.keys())
        for i, s in enumerate(samples):
            if base_name(s) not in paired_bases:
                new_gts.append(gts[i])

        out_fh.write("\t".join(fixed + new_gts) + "\n")

    if args.input != "-":
        in_fh.close()
    if args.output != "-":
        out_fh.close()


if __name__ == "__main__":
    main()
