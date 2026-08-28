#!/usr/bin/env python
"""
Aggregate scratch/refbias/*/refbias_result.json into results/refbias_results.tsv
(long form) and results/refbias.md (headline tables + expectation tests). See
/home/zrm22/.claude/plans/dreamy-booping-sutton.md.

Usage: refbias_report.py [--scratch-root PATH] [--out-tsv PATH] [--out-md PATH]
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import refbias_parse as rp  # noqa: E402
import simval_paths as P  # noqa: E402

SCRATCH_ROOT = P.GRITS_WORKDIR / "scratch/refbias"
RESULTS_DIR = P.GRITS_WORKDIR / "results"

# The 5 founders this corpus ever uses as a true in-index parent. The other
# 20 index founders are NEVER a read set's true source anywhere in the
# corpus -- they are the null/background distribution B73 is judged against.
TRUE_PARENT_FOUNDERS = {"B73", "Oh43", "Il14H", "B97", "CML103"}
NEVER_TRUE_FOUNDERS = [n for n in rp.PANEL_ORDER if n not in TRUE_PARENT_FOUNDERS]


def load_results(scratch_root):
    results = []
    for d in sorted(scratch_root.iterdir()):
        f = d / "refbias_result.json"
        if f.exists():
            results.append(json.loads(f.read_text()))
    return results


def write_long_tsv(results, out_path):
    cols = ["dataset_id", "class", "kind", "individual", "coverage", "arm",
            "founder", "gameteIndex", "is_true_parent", "hit_count", "hit_ratio",
            "hit_ratio_of_input", "singleton_ratio", "placement_rate",
            "mean_cardinality", "total_input_reads", "total_unique_counts"]
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            parents = set(r["parents"]) if r["parents"] else set()
            for gi, founder in enumerate(r["gamete_names"]):
                row = [
                    r["dataset_id"], r["class"], r["kind"], r["individual"],
                    r["coverage"], r["arm"], founder, gi,
                    int(founder in parents),
                    r["gamete_totals"][founder],
                    f"{r['hit_ratio'][founder]:.6f}",
                    f"{r['hit_ratio_of_input'][founder]:.6f}",
                    f"{r['singleton_ratio'][founder]:.6f}",
                    f"{r['placement_rate']:.6f}",
                    f"{r['mean_cardinality']:.4f}",
                    r["total_input_reads"], r["total_unique_counts"],
                ]
                f.write("\t".join(str(x) for x in row) + "\n")


def b73_excess(r):
    """h_B73 / median(h_S over the 20 never-true founders). >1 = B73 elevated
    beyond the never-true background; the de-confounder (refbias_mash.py)
    is what determines whether that excess is real relatedness or bias."""
    bg = [r["hit_ratio"][n] for n in NEVER_TRUE_FOUNDERS]
    med = statistics.median(bg)
    return r["hit_ratio"]["B73"] / med if med else float("nan")


def section_placement_rates(results, out):
    out.write("## Placement rate (fraction of input reads producing output)\n\n")
    out.write("| dataset | individual | arm | placement_rate |\n|---|---|---|---:|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        out.write(f"| {r['dataset_id']} | {r['individual']} | {r['arm']} | "
                   f"{r['placement_rate']:.4f} |\n")
    out.write("\n")


def section_b73_excess(results, out):
    out.write("## B73 excess (hit_ratio[B73] / median(hit_ratio) over the 20 "
               "founders that are never a true parent in this corpus)\n\n")
    out.write("A value near 1 means B73 sits inside the background spread; "
               "a large value means B73 is elevated beyond every founder that "
               "provably contributes no reads. NOT yet relatedness-corrected "
               "(see refbias_mash.py for that).\n\n")
    out.write("| dataset | individual | arm | B73_excess | B73_hit_ratio | "
               "background_median | background_spread(min-max) |\n"
               "|---|---|---|---:|---:|---:|---|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        bg = [r["hit_ratio"][n] for n in NEVER_TRUE_FOUNDERS]
        out.write(f"| {r['dataset_id']} | {r['individual']} | {r['arm']} | "
                   f"{b73_excess(r):.3f} | {r['hit_ratio']['B73']:.4f} | "
                   f"{statistics.median(bg):.4f} | {min(bg):.4f}-{max(bg):.4f} |\n")
    out.write("\n")


def section_in_index_controls(results, out):
    out.write("## In-index inbred lines: does the true founder dominate?\n\n")
    out.write("| dataset | individual(=true founder) | arm | true_founder_hit_ratio | "
               "rank_among_25 | runner_up | runner_up_hit_ratio | margin |\n"
               "|---|---|---|---:|---:|---|---:|---:|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        if r["kind"] != "inbred" or r["class"] != "indexed":
            continue
        true_founder = r["individual"]
        ranked = sorted(r["hit_ratio"].items(), key=lambda kv: -kv[1])
        rank = next(i for i, (n, _) in enumerate(ranked, 1) if n == true_founder)
        true_ratio = r["hit_ratio"][true_founder]
        runner_up_name, runner_up_ratio = (ranked[0] if ranked[0][0] != true_founder else ranked[1])
        out.write(f"| {r['dataset_id']} | {true_founder} | {r['arm']} | "
                   f"{true_ratio:.4f} | {rank} | {runner_up_name} | "
                   f"{runner_up_ratio:.4f} | {true_ratio - runner_up_ratio:.4f} |\n")
    out.write("\n")


def section_out_negative_controls(results, out):
    out.write("## Held-out inbred lines: negative control (no true founder in "
               "index -- profile should be flat, no spike)\n\n")
    out.write("| dataset | individual | arm | top_founder | top_hit_ratio | "
               "spread(max/min) |\n|---|---|---|---|---:|---:|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        if r["kind"] != "inbred" or r["class"] != "heldout":
            continue
        ranked = sorted(r["hit_ratio"].items(), key=lambda kv: -kv[1])
        top_name, top_ratio = ranked[0]
        vals = list(r["hit_ratio"].values())
        spread = max(vals) / min(vals) if min(vals) else float("inf")
        out.write(f"| {r['dataset_id']} | {r['individual']} | {r['arm']} | "
                   f"{top_name} | {top_ratio:.4f} | {spread:.3f}x |\n")
    out.write("\n")


def section_hybrid_ril(results, out):
    out.write("## Hybrid / RIL: parent-ratio check\n\n")
    out.write("Hybrid expectation: ~50/50 between the two true parents (0 breakpoints, "
               "whole-genome 1:1 mix). RIL expectation is dataset-specific (~40 "
               "breakpoints per haplotype) -- exact per-individual ratio needs the "
               "segment-length derivation (not computed by this script; see the plan's "
               "Phase 3). Only reported here for individuals where BOTH parents are "
               "indexed founders (IDX-HYB/IDX-RIL) or the MIX case below -- OUT-HYB/"
               "OUT-RIL have no true-parent signal to check by construction.\n\n")
    out.write("| dataset | individual | arm | parentA | parentB | "
               "A_hit_ratio | B_hit_ratio | A/(A+B) |\n"
               "|---|---|---|---|---|---:|---:|---:|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        if r["kind"] not in ("hybrid", "ril") or not r["parents"] or len(r["parents"]) != 2:
            continue
        pa, pb = r["parents"]
        if pa not in rp.PANEL_ORDER or pb not in rp.PANEL_ORDER:
            continue  # at least one parent held out -- not a same-arm ratio check
        ra, rb = r["hit_ratio"][pa], r["hit_ratio"][pb]
        frac = ra / (ra + rb) if (ra + rb) else float("nan")
        out.write(f"| {r['dataset_id']} | {r['individual']} | {r['arm']} | {pa} | {pb} | "
                   f"{ra:.4f} | {rb:.4f} | {frac:.3f} |\n")
    out.write("\n")


def section_mix(results, out):
    out.write("## MIX (one indexed parent x one held-out parent): where does the "
               "held-out half's signal go?\n\n")
    out.write("The indexed parent has a true home in the index; the held-out parent "
               "does not. This isolates reference-bias fallback behavior with a "
               "same-run internal control -- half the reads in this exact file have "
               "nowhere true to go.\n\n")
    out.write("| dataset | individual | arm | indexed_parent | indexed_hit_ratio | "
               "held_out_parent | B73_hit_ratio | B73_excess |\n"
               "|---|---|---|---|---:|---|---:|---:|\n")
    for r in sorted(results, key=lambda r: (r["dataset_id"], r["individual"], r["arm"])):
        if r["class"] != "mixed" or not r["parents"] or len(r["parents"]) != 2:
            continue
        pa, pb = r["parents"]
        idx_parent = pa if pa in rp.PANEL_ORDER else pb
        out_parent = pb if idx_parent == pa else pa
        out.write(f"| {r['dataset_id']} | {r['individual']} | {r['arm']} | "
                   f"{idx_parent} | {r['hit_ratio'][idx_parent]:.4f} | {out_parent} | "
                   f"{r['hit_ratio']['B73']:.4f} | {b73_excess(r):.3f} |\n")
    out.write("\n")


def section_mash_residual(out):
    mash_tsv = RESULTS_DIR / "refbias_mash_residual.tsv"
    if not mash_tsv.exists():
        return
    out.write("## De-confounded B73 bias estimate (mash-relatedness-corrected)\n\n")
    out.write("For each read set's true source assembly, fit hit_ratio ~ mash-similarity "
               "across the 24 non-B73 founders, then evaluate the residual "
               "(actual - predicted) at B73's own similarity. Near-zero means B73's "
               "elevation is fully explained by real sequence relatedness; a positive "
               "residual is what remains after relatedness is accounted for -- the "
               "bias estimate, in hit_ratio units. See refbias_mash.py.\n\n")
    with open(mash_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(float(r["b73_residual"]))
    out.write("| arm | n | mean_residual | min | max |\n|---|---:|---:|---:|---:|\n")
    for arm, vals in by_arm.items():
        out.write(f"| {arm} | {len(vals)} | {sum(vals)/len(vals):.4f} | "
                   f"{min(vals):.4f} | {max(vals):.4f} |\n")
    out.write("\n(IDX-INBRED/B73 rows compare B73 against itself -- mash similarity "
               "saturates at 1.0, outside the fitted range of the other 24 points, so "
               "that one row's residual is a regression-extrapolation artifact, not "
               "bias; it is included above for completeness but excluded below.)\n\n")
    non_self = [float(r["b73_residual"]) for r in rows if r["individual"] != "B73"]
    by_arm_ns = defaultdict(list)
    for r in rows:
        if r["individual"] != "B73":
            by_arm_ns[r["arm"]].append(float(r["b73_residual"]))
    out.write("Excluding B73's own read set:\n\n")
    out.write("| arm | n | mean_residual | min | max |\n|---|---:|---:|---:|---:|\n")
    for arm, vals in by_arm_ns.items():
        out.write(f"| {arm} | {len(vals)} | {sum(vals)/len(vals):.4f} | "
                   f"{min(vals):.4f} | {max(vals):.4f} |\n")
    out.write("\n")


def section_reciprocity(results, out):
    out.write("## Reciprocity: 5x5 matrix over in-index inbreds (sequence sharing is "
               "symmetric; algorithmic bias need not be)\n\n")
    for arm in sorted(set(r["arm"] for r in results)):
        idx_inbred = {r["individual"]: r for r in results
                      if r["kind"] == "inbred" and r["class"] == "indexed" and r["arm"] == arm}
        founders = sorted(idx_inbred.keys())
        if not founders:
            continue
        out.write(f"### arm={arm}\n\n")
        out.write("row = true source, column = hit_ratio credited to that founder\n\n")
        out.write("| source (row) / credited (col) | " + " | ".join(founders) + " |\n")
        out.write("|---" * (len(founders) + 1) + "|\n")
        for src in founders:
            vals = [f"{idx_inbred[src]['hit_ratio'][c]:.4f}" for c in founders]
            out.write(f"| **{src}** | " + " | ".join(vals) + " |\n")
        out.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-root", default=str(SCRATCH_ROOT))
    ap.add_argument("--out-tsv", default=str(RESULTS_DIR / "refbias_results.tsv"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "refbias.md"))
    args = ap.parse_args()

    results = load_results(Path(args.scratch_root))
    if not results:
        raise SystemExit(f"no refbias_result.json files found under {args.scratch_root}")
    print(f"loaded {len(results)} (individual, arm) results")

    write_long_tsv(results, args.out_tsv)
    print(f"wrote {args.out_tsv}")

    with open(args.out_md, "w") as out:
        out.write("# Reference-bias eval: refmap vs chain (PAV), per-index-sample "
                   "read attribution\n\n")
        out.write(f"{len(results)} (individual, arm) results, 0.1x coverage rung. "
                   "See /home/zrm22/.claude/plans/dreamy-booping-sutton.md for full "
                   "methodology and caveats (esp. the differing carrier/occurrence "
                   "caps between the refmap and chain binaries, and that B73's "
                   "count is partly definitional in both arms).\n\n")
        section_mash_residual(out)
        section_placement_rates(results, out)
        section_b73_excess(results, out)
        section_in_index_controls(results, out)
        section_out_negative_controls(results, out)
        section_hybrid_ril(results, out)
        section_mix(results, out)
        section_reciprocity(results, out)
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
