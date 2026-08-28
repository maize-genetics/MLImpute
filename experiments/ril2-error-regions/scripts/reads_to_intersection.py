#!/usr/bin/env python
"""
Item 3 of the RIL2 error-region diagnostics
(/home/zrm22/.claude/plans/wondrous-discovering-octopus.md): "how many
reads do we go until the intersection set goes to 1 (target genome)".

Works entirely from the existing raw.ps4g for
IDX-RIL2__Oh43xIl14H__0.5x__binsize1 -- the exact run the 210 error regions
were derived from -- with no new refmap run. Per-read gameteSet is
destroyed for EXACT reads in refmap's own raw.tsv (computed, fed to the
PS4G accumulator, never emitted -- see the plan), but PS4G preserves
within-read co-occurrence and only collapses IDENTICAL (contig,bin,
gameteSet) reads into one row with a count. Since re-intersecting a set
with an identical copy of itself is a no-op, expanding each row to `count`
physical read-slots and shuffling THOSE is exactly equivalent to true
per-read intersection tracking -- not an approximation.

For each region: gather PS4G rows whose bp falls inside it, expand to
individual reads, and over T random read orders track:
  - STRICT intersection of gameteSets -- the read index where |I| first
    hits 1 (and whether that founder is correct), or 0 (collapsed to
    nothing, which one bad/PAV read can cause and is expected to be common).
  - TOLERANT intersection -- founders present in >=98% of reads seen so
    far (after a minimum of 3 reads, to avoid a trivial single-read
    "100% support" from resolving nothing). Requested because strict
    intersection is fragile to one bad read; this is the number the
    question is really pointing at.
"""
import argparse
import bisect
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import pandas as pd  # noqa: E402
import simval_paths as P  # noqa: E402
from simval_oracle_bed import build_ril_mosaics  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
PARENT_A, PARENT_B = "Oh43", "Il14H"
RUN_TAG = "binsize1"
COVERAGE = "0.5"
N_TRIALS = 30
TOLERANT_THRESHOLD = 0.98
TOLERANT_MIN_READS = 3


def load_ps4g_index(ps4g_path):
    """Returns per-chrom sorted-by-position (positions, gameteset_str, count) arrays."""
    df = pd.read_csv(ps4g_path, sep="\t", comment="#")
    df = df.sort_values(["refContig", "refPosBinned"]).reset_index(drop=True)
    by_chrom = {}
    for chrom, g in df.groupby("refContig", sort=False):
        by_chrom[chrom] = (
            g["refPosBinned"].to_numpy(),
            g["gameteSet"].tolist(),
            g["count"].to_numpy(),
        )
    return by_chrom


def rows_in_region(chrom, start, end, ps4g_index):
    if chrom not in ps4g_index:
        return [], []
    positions, gamesets, counts = ps4g_index[chrom]
    lo = bisect.bisect_left(positions, start)
    hi = bisect.bisect_left(positions, end)
    return gamesets[lo:hi], counts[lo:hi]


def to_bitmask(gameteset_str):
    mask = 0
    for tok in gameteset_str.split(","):
        mask |= 1 << int(tok)
    return mask


def popcount_list(mask, n=25):
    return [i for i in range(n) if mask & (1 << i)]


def find_true_founder_label(chrom, start, end, mosaic_h1, label_to_name):
    segs = mosaic_h1.get(chrom)
    if not segs:
        return None, None
    starts = [s[0] for s in segs]
    breaks_inside = [s for s in starts if start < s < end]
    if breaks_inside:
        return None, True  # spans a true breakpoint -- no single founder
    i = bisect.bisect_right(starts, start) - 1
    label = segs[max(0, i)][2]
    return label_to_name[label], False


def simulate_region(gamesets, counts, true_founder_idx, rng, n_trials):
    """Expand to individual reads once (order-independent), then shuffle the
    index array per trial -- avoids re-materializing the expansion T times."""
    masks = [to_bitmask(g) for g in gamesets]
    expanded = []
    for i, c in enumerate(counts):
        expanded.extend([i] * int(c))
    n_reads = len(expanded)
    if n_reads == 0:
        return None

    strict_converge_at, strict_correct, strict_collapsed_at = [], [], []
    tolerant_converge_at, tolerant_correct = [], []

    for _ in range(n_trials):
        order = expanded[:]
        rng.shuffle(order)

        strict_mask = None
        strict_done = False
        collapsed_done = False
        support = [0] * 25
        tol_done = False

        for read_idx, row_i in enumerate(order, start=1):
            m = masks[row_i]

            if not strict_done:
                strict_mask = m if strict_mask is None else (strict_mask & m)
                bits = bin(strict_mask).count("1")
                if bits == 0 and not collapsed_done:
                    strict_collapsed_at.append(read_idx)
                    collapsed_done = True
                if bits == 1:
                    strict_converge_at.append(read_idx)
                    winner = popcount_list(strict_mask)[0]
                    strict_correct.append(winner == true_founder_idx if true_founder_idx is not None else None)
                    strict_done = True

            if not tol_done:
                for b in popcount_list(m):
                    support[b] += 1
                if read_idx >= TOLERANT_MIN_READS:
                    at_threshold = [f for f in range(25) if support[f] / read_idx >= TOLERANT_THRESHOLD]
                    if len(at_threshold) == 1:
                        tolerant_converge_at.append(read_idx)
                        tolerant_correct.append(
                            at_threshold[0] == true_founder_idx if true_founder_idx is not None else None)
                        tol_done = True

            if strict_done and tol_done:
                break

    return {
        "n_reads_total": n_reads,
        "n_rows": len(gamesets),
        "strict_median_reads_to_1": st.median(strict_converge_at) if strict_converge_at else None,
        "strict_frac_converged": len(strict_converge_at) / n_trials,
        "strict_frac_correct_given_converged": (
            sum(1 for c in strict_correct if c) / len(strict_correct) if strict_correct and true_founder_idx is not None
            else None),
        "strict_frac_collapsed_to_empty": len(strict_collapsed_at) / n_trials,
        "tolerant_median_reads_to_1": st.median(tolerant_converge_at) if tolerant_converge_at else None,
        "tolerant_frac_converged": len(tolerant_converge_at) / n_trials,
        "tolerant_frac_correct_given_converged": (
            sum(1 for c in tolerant_correct if c) / len(tolerant_correct) if tolerant_correct and true_founder_idx is not None
            else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-json", default="results/ril2_error_regions/error_regions_full.json")
    ap.add_argument("--background-json", default="results/ril2_error_regions/background_regions_full.json")
    ap.add_argument("--out-json", default="results/ril2_error_regions/reads_to_intersection.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--pilot", type=int, default=0)
    cli = ap.parse_args()

    import json
    error_regions = json.loads(Path(cli.error_json).read_text())
    bg_regions = json.loads(Path(cli.background_json).read_text())
    if cli.pilot:
        error_regions = error_regions[: cli.pilot]
        bg_regions = bg_regions[: cli.pilot]
    all_regions = [(r, "error") for r in error_regions] + [(r, "background") for r in bg_regions]

    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{COVERAGE}x__{RUN_TAG}"
    ps4g_path = outdir / "raw.ps4g"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    gamete_names = pd.read_csv(gametes_path, sep="\t").sort_values("gameteIndex")["sampleName"].tolist()
    name_to_idx = {n: i for i, n in enumerate(gamete_names)}

    print(f"loading {ps4g_path} ...")
    ps4g_index = load_ps4g_index(ps4g_path)
    print(f"loaded, chroms: {sorted(ps4g_index.keys())}")

    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", PARENT_A, PARENT_B)

    rng = random.Random(cli.seed)
    results = []
    for r, kind in all_regions:
        true_founder = r.get("true_founder")
        spans_breakpoint = None
        if true_founder is None:
            true_founder, spans_breakpoint = find_true_founder_label(
                r["chrom"], r["start"], r["end"], mosaic_h1, label_to_name)
        true_idx = name_to_idx.get(true_founder) if true_founder else None

        gamesets, counts = rows_in_region(r["chrom"], r["start"], r["end"], ps4g_index)
        sim = simulate_region(gamesets, counts, true_idx, rng, cli.n_trials)

        row = {"chrom": r["chrom"], "start": r["start"], "end": r["end"], "width": r["width"],
               "kind": kind, "true_founder": true_founder, "spans_breakpoint": spans_breakpoint}
        row.update(sim if sim else {"n_reads_total": 0})
        results.append(row)

    Path(cli.out_json).write_text(json.dumps(results, indent=1))
    n_empty = sum(1 for r in results if r.get("n_reads_total", 0) == 0)
    n_span = sum(1 for r in results if r.get("spans_breakpoint"))
    print(f"\n{len(results)} regions, {n_empty} with zero PS4G rows, "
          f"{n_span} background regions span a true breakpoint (dropped from correctness stat)")

    err_med = [r.get("strict_median_reads_to_1") for r in results if r["kind"] == "error" and r.get("strict_median_reads_to_1")]
    bg_med = [r.get("strict_median_reads_to_1") for r in results if r["kind"] == "background" and r.get("strict_median_reads_to_1")]
    print(f"\nstrict median reads-to-1: error median={st.median(err_med) if err_med else None} "
          f"(n={len(err_med)})  background median={st.median(bg_med) if bg_med else None} (n={len(bg_med)})")

    err_tol = [r.get("tolerant_median_reads_to_1") for r in results if r["kind"] == "error" and r.get("tolerant_median_reads_to_1")]
    bg_tol = [r.get("tolerant_median_reads_to_1") for r in results if r["kind"] == "background" and r.get("tolerant_median_reads_to_1")]
    print(f"tolerant median reads-to-1: error median={st.median(err_tol) if err_tol else None} "
          f"(n={len(err_tol)})  background median={st.median(bg_tol) if bg_tol else None} (n={len(bg_tol)})")

    try:
        from scipy import stats as sps
        if err_med and bg_med:
            u, p = sps.mannwhitneyu(err_med, bg_med, alternative="two-sided")
            print(f"  strict Mann-Whitney p={p:.4g}")
        if err_tol and bg_tol:
            u, p = sps.mannwhitneyu(err_tol, bg_tol, alternative="two-sided")
            print(f"  tolerant Mann-Whitney p={p:.4g}")
    except ImportError:
        print("  (scipy unavailable)")

    err_correct = [r.get("strict_frac_correct_given_converged") for r in results
                   if r["kind"] == "error" and r.get("strict_frac_correct_given_converged") is not None]
    print(f"\nstrict-converged error regions: mean P(converges to TRUE founder)="
          f"{st.mean(err_correct):.3f}" if err_correct else "\n(no error regions with a defined true founder)")

    print(f"\nwrote {cli.out_json}")


if __name__ == "__main__":
    main()
