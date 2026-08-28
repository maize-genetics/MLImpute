#!/usr/bin/env python
"""
Does the anchor_density_vs_error.py classification (founder_specific_dropout
/ general_difficulty / unexplained_by_anchor_density, and the 65/22/13 bp
split found for Oh43xIl14H 0.5x) generalize to other founder pairs and
other coverages, or is it specific to that one sample?

Reuses extract_error_regions.py's split-at-true-breakpoint logic and
sample_background_regions.py's width-matched sampling, generalized to any
IDX-RIL2 pair, plus anchor_density_vs_error.py's classification against the
already-built (pair/coverage-independent) results/founder_touchdown_map/
matrix -- no new refmap runs, no new alignment, everything reused.

Uses the `unfiltered-bin` tag uniformly across ALL pairs/coverages (the
only tag available for the 4 non-Oh43xIl14H pairs) -- NOTE this makes the
Oh43xIl14H numbers here differ slightly from DIAGNOSTICS.md's headline
figures, which used the `binsize1` tag; both are valid, just not identical
runs. B73xCML103 and B73xOh43 have B73 as one parent -- segments truly
labeled "B73" are skipped (B73 is deliberately excluded from the founder
touchdown matrix, see build_touchdown_matrix.py), so those two pairs have
roughly half as many scoreable error regions as their raw error count.
"""
import json
import random
from pathlib import Path

import numpy as np
from scipy import stats as sps

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from simval_oracle_bed import build_ril_mosaics  # noqa: E402
from extract_error_regions import extract_wrong_intervals  # noqa: E402
import simval_paths as P  # noqa: E402

PAIRS = ["Oh43xIl14H", "B73xCML103", "B73xOh43", "B97xCML103", "Il14HxB97"]
COVERAGES = ["0.01", "0.1", "0.5", "1.0", "2.0"]
TAG = "unfiltered-bin"
N_BACKGROUND = 1000
RATIO_CUTOFF = 0.5

FAI = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai"


def load_chrom_lengths():
    lengths = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            if name.startswith("chr") and name[3:].isdigit():
                lengths[name] = int(length)
    return lengths


def sample_background(errors, chrom_lengths, n, seed):
    widths = [max(1, e["width"]) for e in errors]
    chroms = list(chrom_lengths.keys())
    total_len = sum(chrom_lengths.values())
    weights = [chrom_lengths[c] / total_len for c in chroms]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chrom = rng.choices(chroms, weights=weights, k=1)[0]
        width = rng.choice(widths)
        s = rng.randint(0, max(0, chrom_lengths[chrom] - width))
        out.append({"chrom": chrom, "start": s, "end": s + width})
    return out


def find_true_founder(chrom, start, end, mosaic_h1, label_to_name):
    import bisect
    segs = mosaic_h1.get(chrom)
    if not segs:
        return None
    starts = [s[0] for s in segs]
    if any(start < s < end for s in starts):
        return None  # spans a true breakpoint
    i = bisect.bisect_right(starts, start) - 1
    return label_to_name[segs[max(0, i)][2]]


def bins_for(off, bin_size, chrom, start, end):
    o = off[chrom]
    lo = o + start // bin_size
    hi = o + (end - 1) // bin_size
    return list(range(lo, hi + 1))


def score_regions(regions, mosaic_h1, label_to_name, matrix, fidx, median_per_bin, off, bin_size):
    rows = []
    for r in regions:
        tf = find_true_founder(r["chrom"], r["start"], r["end"], mosaic_h1, label_to_name)
        if not tf or tf not in fidx:
            continue
        fi = fidx[tf]
        b_idxs = bins_for(off, bin_size, r["chrom"], r["start"], r["end"])
        counts = matrix[fi, b_idxs]
        meds = median_per_bin[b_idxs]
        ratios = (counts + 0.5) / (meds + 0.5)
        rows.append({"chrom": r["chrom"], "start": r["start"], "end": r["end"], "true_founder": tf,
                     "min_ratio": float(ratios.min()), "min_count": int(counts.min())})
    return rows


def main():
    td = json.loads(Path("results/founder_touchdown_map/touchdown_matrix_100kb.json").read_text())
    matrix = np.array(td["matrix"], dtype=np.int64)
    founders = td["founders"]
    fidx = {f: i for i, f in enumerate(founders)}
    bin_size = td["bin_size"]
    off = td["bin_offset"]
    median_per_bin = np.median(matrix, axis=0)
    chrom_lengths = load_chrom_lengths()

    results = []
    for pair in PAIRS:
        parent_a, parent_b = pair.split("x")
        mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", parent_a, parent_b)
        for coverage in COVERAGES:
            bed_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{pair}__{coverage}x__{TAG}" / "bed"
            sample = f"{pair}_ril2_{TAG}"
            if not bed_dir.exists():
                print(f"SKIP {pair} {coverage}x: no {bed_dir}")
                continue
            wrong_intervals, wrong_bp, total_bp = extract_wrong_intervals(bed_dir, sample, mosaic_h1, label_to_name)
            if len(wrong_intervals) < 5:
                print(f"{pair} {coverage}x: only {len(wrong_intervals)} error intervals, skipping (too few)")
                continue

            bg_regions = sample_background(wrong_intervals, chrom_lengths, N_BACKGROUND, seed=hash((pair, coverage)) & 0xffffffff)

            err_rows = score_regions(wrong_intervals, mosaic_h1, label_to_name, matrix, fidx, median_per_bin, off, bin_size)
            bg_rows = score_regions(bg_regions, mosaic_h1, label_to_name, matrix, fidx, median_per_bin, off, bin_size)
            if len(err_rows) < 5 or len(bg_rows) < 20:
                print(f"{pair} {coverage}x: too few scoreable rows (err={len(err_rows)} bg={len(bg_rows)}), skipping")
                continue

            count_cutoff = float(np.median([r["min_count"] for r in bg_rows]))
            bp_by_mech = {"founder_specific_dropout": 0, "general_difficulty": 0, "unexplained_by_anchor_density": 0}
            n_by_mech = {k: 0 for k in bp_by_mech}
            for r in err_rows:
                ratio_low = r["min_ratio"] < RATIO_CUTOFF
                count_low = r["min_count"] < count_cutoff
                mech = "founder_specific_dropout" if (ratio_low and count_low) else \
                       "general_difficulty" if count_low else "unexplained_by_anchor_density"
                bp = r["end"] - r["start"]
                bp_by_mech[mech] += bp
                n_by_mech[mech] += 1
            total_err_bp = sum(bp_by_mech.values())

            e_count = [r["min_count"] for r in err_rows]
            b_count = [r["min_count"] for r in bg_rows]
            _, p_count = sps.mannwhitneyu(e_count, b_count, alternative="two-sided")

            row = {
                "pair": pair, "coverage": coverage,
                "n_error_intervals": len(wrong_intervals), "n_scoreable_error": len(err_rows),
                "error_pct_bp": round(wrong_bp / total_bp * 100, 4),
                "total_error_bp": total_err_bp, "count_cutoff": count_cutoff,
                "p_mannwhitney_count": float(p_count),
            }
            for mech in bp_by_mech:
                row[f"{mech}_n"] = n_by_mech[mech]
                row[f"{mech}_bp"] = bp_by_mech[mech]
                row[f"{mech}_pct_bp"] = round(bp_by_mech[mech] / total_err_bp * 100, 1) if total_err_bp else None
            results.append(row)
            print(f"{pair:12s} {coverage:>5s}x  n_err={len(err_rows):4d}  "
                  f"dropout={row['founder_specific_dropout_pct_bp']:5.1f}%  "
                  f"general={row['general_difficulty_pct_bp']:5.1f}%  "
                  f"unexplained={row['unexplained_by_anchor_density_pct_bp']:5.1f}%  "
                  f"p={p_count:.2g}")

    Path("results/ril2_error_regions/multi_pair_anchor_generalization.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote results/ril2_error_regions/multi_pair_anchor_generalization.json ({len(results)} rows)")


if __name__ == "__main__":
    main()
