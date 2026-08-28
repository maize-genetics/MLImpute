#!/usr/bin/env python
"""
Annotate a list of genomic regions (error intervals, or a random background
sample of the same shape) with: distance to nearest N-gap, and overlap with
an Il14H PAV-proxy region (gap in AnchorWave alignment coverage). SSR
content is handled separately by local_ssr_content.py (needs samtools+trf).
"""
import argparse
import bisect
import json
from pathlib import Path


def load_bed(path):
    by_chrom = {}
    with open(path) as f:
        for line in f:
            chrom, s, e = line.split()[:3]
            by_chrom.setdefault(chrom, []).append((int(s), int(e)))
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom


def nearest_gap_distance(chrom, mid, gaps_by_chrom):
    intervals = gaps_by_chrom.get(chrom, [])
    if not intervals:
        return None
    starts = [s for s, _ in intervals]
    i = bisect.bisect_left(starts, mid)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(intervals):
            s, e = intervals[j]
            if s <= mid <= e:
                d = 0
            else:
                d = min(abs(mid - s), abs(mid - e))
            if best is None or d < best:
                best = d
    return best


def overlaps_any(chrom, s, e, intervals_by_chrom):
    intervals = intervals_by_chrom.get(chrom, [])
    starts = [x[0] for x in intervals]
    i = bisect.bisect_right(starts, e)
    for j in range(max(0, i - 3), min(len(intervals), i + 1)):
        gs, ge = intervals[j]
        if gs < e and s < ge:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions-json", required=True)
    ap.add_argument("--n-gaps-bed", required=True)
    ap.add_argument("--pav-proxy-bed", required=True)
    ap.add_argument("--out-json", required=True)
    cli = ap.parse_args()

    regions = json.loads(Path(cli.regions_json).read_text())
    n_gaps = load_bed(cli.n_gaps_bed)
    pav = load_bed(cli.pav_proxy_bed)

    for r in regions:
        mid = (r["start"] + r["end"]) // 2
        r["dist_to_n_gap"] = nearest_gap_distance(r["chrom"], mid, n_gaps)
        r["in_il14h_pav_proxy"] = overlaps_any(r["chrom"], r["start"], r["end"], pav)

    Path(cli.out_json).write_text(json.dumps(regions, indent=1))
    n_pav = sum(1 for r in regions if r["in_il14h_pav_proxy"])
    dists = [r["dist_to_n_gap"] for r in regions if r["dist_to_n_gap"] is not None]
    print(f"{len(regions)} regions: {n_pav} ({n_pav/len(regions):.1%}) in Il14H PAV-proxy region")
    if dists:
        dists.sort()
        print(f"dist_to_n_gap: median={dists[len(dists)//2]:,} "
              f"min={dists[0]:,} p90={dists[int(len(dists)*0.9)]:,}")
    print(f"wrote {cli.out_json}")


if __name__ == "__main__":
    main()
