#!/usr/bin/env python
"""
Item 2 of the RIL2 error-region diagnostics
(/home/zrm22/.claude/plans/wondrous-discovering-octopus.md): "non-continuity
stat for repetitiveness, redundancy and structural changes".

Two independent families, both error-vs-background, both reusing existing
data with no new generation:

Structural (from il14h_covered.merged.bed + b73_n_gaps.bed, already built
for the N-gap/PAV-proxy annotation pass): how fragmented is the region's
Il14H-vs-B73 alignment coverage. A colinear region has ~0 covered/uncovered
transitions; a structurally rearranged one is shredded into many short
segments.

Repetitiveness/redundancy (reuses region_occ_stats.json's per-tile
occurrence series -- no new ropebwt3 query): how choppy is the
pangenome-occurrence signal along the region, as opposed to region_occ_
stats.py's item-1 summary stats which collapse it to one number per region.
"""
import argparse
import bisect
import json
import math
import statistics as st
from pathlib import Path


def load_bed(path):
    by_chrom = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom, s, e = parts[0], int(parts[1]), int(parts[2])
            by_chrom.setdefault(chrom, []).append((s, e))
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom


def segments_in_region(intervals, start, end):
    """Intervals (sorted, non-overlapping) clipped to [start,end)."""
    if not intervals:
        return []
    starts = [s for s, _ in intervals]
    i = max(0, bisect.bisect_right(starts, end) - 1)
    out = []
    # scan backward from i then forward is unnecessary since starts sorted and
    # intervals non-overlapping -- walk left to catch any interval starting before
    # `start` that still overlaps it, then walk right until past `end`.
    j = i
    while j >= 0 and intervals[j][1] > start:
        j -= 1
    j += 1
    while j < len(intervals) and intervals[j][0] < end:
        s, e = intervals[j]
        cs, ce = max(s, start), min(e, end)
        if ce > cs:
            out.append((cs, ce))
        j += 1
    return out


def cv(values):
    if len(values) < 2:
        return None
    m = st.mean(values)
    return (st.pstdev(values) / m) if m > 0 else None


def structural_stats(chrom, start, end, covered_by_chrom, ngap_by_chrom):
    width = end - start
    covered = segments_in_region(covered_by_chrom.get(chrom, []), start, end)
    covered_bp = sum(e - s for s, e in covered)
    n_breaks = len(covered)  # each clipped covered segment implies a boundary on both sides
    # gaps = complement of covered segments within [start,end)
    gaps = []
    cursor = start
    for s, e in covered:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = e
    if cursor < end:
        gaps.append((cursor, end))
    gap_lens = [e - s for s, e in gaps]
    seg_lens = [e - s for s, e in covered]

    ngaps = segments_in_region(ngap_by_chrom.get(chrom, []), start, end)
    n_gap_bp = sum(e - s for s, e in ngaps)

    return {
        "covered_fraction": round(covered_bp / width, 4) if width else None,
        "n_breaks_per_10kb": round(n_breaks * 10000 / width, 4) if width else None,
        "cv_segment_len": round(cv(seg_lens), 4) if cv(seg_lens) is not None else None,
        "cv_gap_len": round(cv(gap_lens), 4) if cv(gap_lens) is not None else None,
        "longest_uncovered_bp": max(gap_lens) if gap_lens else 0,
        "n_gap_bp_in_region": n_gap_bp,
    }


def repetitiveness_stats(tile_occs):
    if len(tile_occs) < 2:
        return {"occ_jump_rate": None, "occ_log_acf1": None}
    jumps = sum(1 for a, b in zip(tile_occs, tile_occs[1:])
                if (max(a, b) / min(a, b)) > 2 if min(a, b) > 0)
    jump_rate = jumps / (len(tile_occs) - 1)
    logs = [math.log10(max(1, x)) for x in tile_occs]
    m = st.mean(logs)
    x = [v - m for v in logs]
    denom = sum(v * v for v in x)
    acf1 = (sum(a * b for a, b in zip(x[:-1], x[1:])) / denom) if denom > 0 else None
    return {"occ_jump_rate": round(jump_rate, 4), "occ_log_acf1": round(acf1, 4) if acf1 is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ-stats-json", default="results/ril2_error_regions/region_occ_stats.json")
    ap.add_argument("--covered-bed", default="results/ril2_error_regions/il14h_covered.merged.bed")
    ap.add_argument("--n-gaps-bed", default="results/ril2_error_regions/b73_n_gaps.bed")
    ap.add_argument("--out-json", default="results/ril2_error_regions/region_noncontinuity.json")
    cli = ap.parse_args()

    regions = json.loads(Path(cli.occ_stats_json).read_text())
    covered_by_chrom = load_bed(cli.covered_bed)
    ngap_by_chrom = load_bed(cli.n_gaps_bed)

    results = []
    for r in regions:
        row = {"chrom": r["chrom"], "start": r["start"], "end": r["end"], "width": r["width"],
               "kind": r["kind"], "true_founder": r.get("true_founder")}
        row.update(structural_stats(r["chrom"], r["start"], r["end"], covered_by_chrom, ngap_by_chrom))
        row.update(repetitiveness_stats(r.get("_tile_occs", [])))
        results.append(row)

    Path(cli.out_json).write_text(json.dumps(results, indent=1))

    # sanity check (plan step 2): background covered_fraction should be near the
    # genome-wide Il14H covered fraction (~57.6% = 100% - 42.4% PAV-proxy).
    bg = [r["covered_fraction"] for r in results if r["kind"] == "background" and r["covered_fraction"] is not None]
    print(f"background covered_fraction mean={st.mean(bg):.4f} (expect ~0.576 genome-wide) n={len(bg)}")

    err_vals = {k: [r[k] for r in results if r["kind"] == "error" and r[k] is not None]
                for k in ("covered_fraction", "n_breaks_per_10kb", "cv_segment_len", "cv_gap_len",
                          "occ_jump_rate", "occ_log_acf1")}
    bg_vals = {k: [r[k] for r in results if r["kind"] == "background" and r[k] is not None]
               for k in err_vals}
    print(f"\nerror vs background:")
    try:
        from scipy import stats as sps
        for k in err_vals:
            e, b = err_vals[k], bg_vals[k]
            if len(e) < 2 or len(b) < 2:
                continue
            u, p = sps.mannwhitneyu(e, b, alternative="two-sided")
            print(f"  {k}: error(n={len(e)}) median={st.median(e):.4g}  "
                  f"background(n={len(b)}) median={st.median(b):.4g}  Mann-Whitney p={p:.4g}")
    except ImportError:
        print("  (scipy unavailable -- rerun under ml-impute-env)")

    print(f"\nwrote {cli.out_json}")


if __name__ == "__main__":
    main()
