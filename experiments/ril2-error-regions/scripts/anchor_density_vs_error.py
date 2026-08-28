#!/usr/bin/env python
"""
Direct test of the prediction: if a founder's assembly anchors poorly to
B73 at a locus (results/founder_touchdown_map/), does the RIL2 founder-path
decode error rate go up there? Scores the true founder's anchor density at
each of the 210 real error intervals and 1000 background intervals
(results/ril2_error_regions/reads_to_intersection.json already has
chrom/start/end/kind/true_founder for both groups -- true_founder for
background regions is filled via simval_oracle_bed's exact RNG-replayed
mosaic, same source used throughout this diagnostics line of work).

Two metrics, both taken as the worst (min) and average (mean) over every
100kb touchdown-map bin the region overlaps:
  - raw anchor count for the true founder -- does the prediction hold in
    absolute terms?
  - anomaly ratio (count / median-across-all-founders-at-that-bin) -- does
    it hold once "how hard is this locus for everyone" is controlled for?

The two can disagree, and did: raw count is strongly, significantly lower
in error regions (confirms the prediction), but the ratio is barely
different between error and background (the founder-specific-dropout
story is not the dominant mechanism population-wide) -- see classification
below for the breakdown.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

# min_ratio < RATIO_CUTOFF: true founder notably worse than its peers at this locus.
# min_count < COUNT_CUTOFF: true founder's raw anchor count below the background median
# (COUNT_CUTOFF is set from the background distribution itself, not hardcoded).
RATIO_CUTOFF = 0.5


def bins_for(off, bin_size, chrom, start, end):
    o = off[chrom]
    lo = o + start // bin_size
    hi = o + (end - 1) // bin_size
    return list(range(lo, hi + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions-json", default="results/ril2_error_regions/reads_to_intersection.json")
    ap.add_argument("--touchdown-json", default="results/founder_touchdown_map/touchdown_matrix_100kb.json")
    ap.add_argument("--out-json", default="results/ril2_error_regions/anchor_density_vs_error.json")
    cli = ap.parse_args()

    regions = json.loads(Path(cli.regions_json).read_text())
    td = json.loads(Path(cli.touchdown_json).read_text())
    m = np.array(td["matrix"], dtype=np.int64)
    founders = td["founders"]
    fidx = {f: i for i, f in enumerate(founders)}
    bin_size = td["bin_size"]
    off = td["bin_offset"]
    median_per_bin = np.median(m, axis=0)

    skipped = 0
    rows = []
    for r in regions:
        tf = r.get("true_founder")
        if not tf or tf not in fidx:
            skipped += 1
            continue
        fi = fidx[tf]
        b_idxs = bins_for(off, bin_size, r["chrom"], r["start"], r["end"])
        counts = m[fi, b_idxs]
        meds = median_per_bin[b_idxs]
        ratios = (counts + 0.5) / (meds + 0.5)
        rows.append({
            "chrom": r["chrom"], "start": r["start"], "end": r["end"],
            "kind": r["kind"], "true_founder": tf,
            "min_ratio": round(float(ratios.min()), 4), "mean_ratio": round(float(ratios.mean()), 4),
            "min_count": int(counts.min()), "mean_count": round(float(counts.mean()), 3),
        })

    err = [r for r in rows if r["kind"] == "error"]
    bg = [r for r in rows if r["kind"] == "background"]
    print(f"{len(rows)} regions scored, {skipped} skipped (no true_founder / spans breakpoint)")
    print(f"error n={len(err)}  background n={len(bg)}")

    tests = {}
    for key in ["min_ratio", "mean_ratio", "min_count", "mean_count"]:
        e = [r[key] for r in err]
        b = [r[key] for r in bg]
        u, p = sps.mannwhitneyu(e, b, alternative="two-sided")
        tests[key] = {"error_median": float(np.median(e)), "background_median": float(np.median(b)), "p": float(p)}
        print(f"{key}: error median={np.median(e):.3f}  background median={np.median(b):.3f}  Mann-Whitney p={p:.4g}")

    # classify each ERROR region by which mechanism (if either) its anchor data fits
    count_cutoff = float(np.median([r["min_count"] for r in bg]))
    for r in err:
        ratio_low = r["min_ratio"] < RATIO_CUTOFF
        count_low = r["min_count"] < count_cutoff
        if ratio_low and count_low:
            r["mechanism"] = "founder_specific_dropout"
        elif count_low:
            r["mechanism"] = "general_difficulty"
        else:
            r["mechanism"] = "unexplained_by_anchor_density"

    counts = {}
    bp_by_mech = {}
    for r in err:
        counts[r["mechanism"]] = counts.get(r["mechanism"], 0) + 1
        bp_by_mech[r["mechanism"]] = bp_by_mech.get(r["mechanism"], 0) + (r["end"] - r["start"])
    total_err_bp = sum(r["end"] - r["start"] for r in err)
    print(f"\nclassification of the 210 error regions (ratio cutoff={RATIO_CUTOFF}, "
          f"count cutoff={count_cutoff:.0f} = background min_count median):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        bp = bp_by_mech[k]
        print(f"  {k}: {v} regions ({v/len(err):.1%})  {bp:,} bp ({bp/total_err_bp:.1%} of error bp, "
              f"avg width {bp/v:,.0f} bp)")
    print(f"  TOTAL: {len(err)} regions, {total_err_bp:,} bp ({total_err_bp/1e6:.2f} Mb)")

    out = {"rows": rows, "tests": tests, "classification_counts": counts,
           "classification_bp": bp_by_mech, "total_error_bp": total_err_bp,
           "ratio_cutoff": RATIO_CUTOFF, "count_cutoff": count_cutoff}
    Path(cli.out_json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {cli.out_json}")


if __name__ == "__main__":
    main()
