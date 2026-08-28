#!/usr/bin/env python
"""
SNP density (per covered bp) for a founder vs B73, from its AnchorWave
gVCF -- a cheap divergence proxy to test whether CML103 is unusually
similar to B73 (which would explain the B73xCML103 mutual-confusion
finding in results/ril2_error_regions/DIAGNOSTICS.md item 5/6) compared
to a founder like Oh43 that shows genuine locus-specific anchor dropout
instead of reference confusion.

Streams `bcftools view -H` once (no full load into Python), counts:
  - snp_count: records where REF is 1bp, the first ALT allele is 1bp and
    not "<NON_REF>", and differs from REF (a real substitution)
  - covered_bp: sum of each record's span (END-POS+1 from INFO if present,
    else 1bp) -- the gVCF's own genome coverage, used as the denominator
    so SNP density is comparable even between founders with different
    overall PAV/coverage extent (e.g. Il14H's ~57.6% coverage vs others).
"""
import argparse
import re
import subprocess
import sys

END_RE = re.compile(rb"(?:^|;)END=(\d+)")


def scan(gvcf_path):
    snp_count = 0
    covered_bp = 0
    proc = subprocess.Popen(["bcftools", "view", "-H", gvcf_path], stdout=subprocess.PIPE)
    for line in proc.stdout:
        parts = line.rstrip(b"\n").split(b"\t")
        pos, ref, alt, info = int(parts[1]), parts[3], parts[4], parts[7]
        m = END_RE.search(info)
        end = int(m.group(1)) if m else pos
        covered_bp += end - pos + 1
        alt0 = alt.split(b",")[0]
        if len(ref) == 1 and len(alt0) == 1 and alt0 != b"<NON_REF>" and alt0 != ref:
            snp_count += 1
    proc.wait()
    return snp_count, covered_bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvcf", required=True, action="append", help="repeatable: path[,label]")
    cli = ap.parse_args()

    for spec in cli.gvcf:
        path, _, label = spec.partition(",")
        label = label or path
        snp_count, covered_bp = scan(path)
        density = snp_count / covered_bp * 1000
        print(f"{label}: snp_count={snp_count:,}  covered_bp={covered_bp:,}  "
              f"density={density:.3f} SNPs/kb covered")


if __name__ == "__main__":
    main()
