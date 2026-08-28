#!/usr/bin/env python
"""
Report for the zero-sequencing-error control
(/home/zrm22/.claude/plans/deep-jumping-salamander.md): collects arms
A (published baseline), B (no-error, -e 0), C (redraw control, -e 0.001)
for IDX-HYB/Oh43xIl14H/0.1x into one TSV/MD, plus the PS4G per-founder
background comparison and, for B/C only (they carry L1 read labels), a
true-source x credited-founder confusion matrix built from raw.tsv.

Usage:
    simval_noerr_report.py \
        --arm-b-dir scratch/simval_eval/IDX-HYB__Oh43xIl14H__0.1x__noerror \
        --arm-c-dir scratch/simval_eval/IDX-HYB__Oh43xIl14H__0.1x__redraw
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import simval_paths as P  # noqa: E402
import refbias_parse as rp  # noqa: E402

BASELINE_ROW_KEY = "IDX-HYB__Oh43xIl14H__0.1x"
BASELINE_OUTDIR = P.SCRATCH_ROOT / BASELINE_ROW_KEY
PANEL_FOUNDERS = rp.PANEL_ORDER  # no underscores in any name -> safe split point
FOUNDER_PREFIX_RE = re.compile("^(" + "|".join(re.escape(n) for n in PANEL_FOUNDERS) + ")_")


# ---------------------------------------------------------------------------
# Metric loaders
# ---------------------------------------------------------------------------

def load_arm_metrics(outdir, label):
    """Baseline's result.json (written by simval_batch.py) is flat; a direct
    simval_eval_one.py --out-json is nested under "align"/"score" keys
    (simval_eval_one.py:412-428). Handle both."""
    result_json = outdir / (f"{label}_result.json" if (outdir / f"{label}_result.json").exists()
                             else "result.json")
    if not result_json.exists():
        cands = list(outdir.glob("*result.json"))
        if not cands:
            raise FileNotFoundError(f"no result json under {outdir}")
        result_json = cands[0]
    data = json.loads(result_json.read_text())
    score = data.get("score", data)
    align = data.get("align", data)
    return {
        "error_rate": score.get("error_rate"),
        "partial_error_rate": score.get("partial_error_rate"),
        "compared_sites": score.get("compared_sites"),
        "align_dropped_idx": align.get("dropped_idx", data.get("align_dropped_idx")),
        "align_het_scale": align.get("het_scale", data.get("align_het_scale")),
        "imputed_records": score.get("imputed_records"),
    }


def load_snprc_metrics(json_path):
    """Mirrors simval_snp_refcall_rescore.summarize()'s exact formulas -- the
    comparator's --json-out never writes snprc_error_rate/partial_error_rate
    directly, only the raw counts these are derived from."""
    if not Path(json_path).exists():
        return {"snprc_error_rate": None, "snprc_partial_error_rate": None,
                "snprc_compared_sites": None}
    data = json.loads(Path(json_path).read_text())
    compared = data.get("snprc_compared_sites", 0)
    error_rate = (1.0 - data["snprc_gt_allele_matches"] / compared) if compared else None
    partial_error_rate = (1.0 - data["snprc_partial_credit_sum"] / compared) if compared else None
    return {
        "snprc_error_rate": error_rate,
        "snprc_partial_error_rate": partial_error_rate,
        "snprc_compared_sites": compared,
    }


# ---------------------------------------------------------------------------
# PS4G aggregate comparison
# ---------------------------------------------------------------------------

def ps4g_background_summary(ps4g_path, true_parents):
    header = rp.parse_ps4g_header(ps4g_path)
    totals = header["gamete_totals"]
    parent_counts = [totals[p] for p in true_parents if p in totals]
    other_counts = [c for name, c in totals.items() if name not in true_parents]
    mean_parent = sum(parent_counts) / len(parent_counts) if parent_counts else float("nan")
    median_other = sorted(other_counts)[len(other_counts) // 2] if other_counts else float("nan")
    return {
        "total_unique_counts": header["total_unique_counts"],
        "gamete_totals": totals,
        "mean_true_parent_count": mean_parent,
        "median_nonparent_count": median_other,
        "background_ratio": median_other / mean_parent if mean_parent else float("nan"),
    }


# ---------------------------------------------------------------------------
# L3 -- true-source x credited-founder confusion matrix (arms B/C only)
# ---------------------------------------------------------------------------

def credited_founders(hit_field):
    """raw.tsv col5, e.g. 'Ki3_chr1:+,CML69_chr1:+,...' or '.' -> set of founder names."""
    if hit_field == "." or not hit_field:
        return set()
    out = set()
    for entry in hit_field.split(","):
        m = FOUNDER_PREFIX_RE.match(entry)
        if m:
            out.add(m.group(1))
    return out


def build_confusion_matrix(raw_tsv_path):
    """Returns (matrix: {true_source: {credited_founder: count}}, status_counts:
    {PLACED/EXACT/MULTI/UNPLACED: count}, per_source_totals: {source: n_reads})."""
    matrix = defaultdict(lambda: defaultdict(int))
    status_counts = defaultdict(int)
    per_source_totals = defaultdict(int)
    unlabeled = 0
    with open(raw_tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            read_name, _length, status, _nhits, hits = parts[0], parts[1], parts[2], parts[3], parts[4]
            status_counts[status] += 1
            if "|" not in read_name:
                unlabeled += 1
                continue
            source = read_name.split("|", 1)[0]
            per_source_totals[source] += 1
            if status == "PLACED":
                for founder in credited_founders(hits):
                    matrix[source][founder] += 1
            else:
                matrix[source][f"_{status}"] += 1
    return (
        {k: dict(v) for k, v in matrix.items()},
        dict(status_counts),
        dict(per_source_totals),
        unlabeled,
    )


def summarize_confusion(matrix, per_source_totals, true_parents):
    """For each true source, self-credit rate among PLACED reads, and mean
    background credit rate to non-self founders."""
    out = {}
    for source in true_parents:
        row = matrix.get(source, {})
        placed_credits = {k: v for k, v in row.items() if not k.startswith("_")}
        total_placed = sum(placed_credits.values())
        self_credit = placed_credits.get(source, 0)
        other = {k: v for k, v in placed_credits.items() if k != source}
        mean_other = sum(other.values()) / len(other) if other else 0.0
        out[source] = {
            "n_reads_from_source": per_source_totals.get(source, 0),
            "status_EXACT": row.get("_EXACT", 0),
            "status_MULTI": row.get("_MULTI", 0),
            "status_UNPLACED": row.get("_UNPLACED", 0),
            "placed_total_credits": total_placed,
            "self_credit": self_credit,
            "self_credit_rate": self_credit / total_placed if total_placed else float("nan"),
            "mean_other_founder_credit": mean_other,
            "background_ratio": mean_other / self_credit if self_credit else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-b-dir", required=True, help="no-error arm outdir")
    ap.add_argument("--arm-c-dir", required=True, help="redraw-control arm outdir")
    ap.add_argument("--arm-b-snprc-json", default=None)
    ap.add_argument("--arm-c-snprc-json", default=None)
    ap.add_argument("--out-tsv", default=str(P.RESULTS_DIR / "simval_noerr_oh43xil14h.tsv"))
    ap.add_argument("--out-md", default=str(P.RESULTS_DIR / "simval_noerr_oh43xil14h.md"))
    args = ap.parse_args()

    true_parents = ["Oh43", "Il14H"]
    arm_b_dir = Path(args.arm_b_dir)
    arm_c_dir = Path(args.arm_c_dir)

    # --- all-sites + SNP+RefCall metrics, all three arms ---
    baseline_json = BASELINE_OUTDIR / "result.json"
    arms = {
        "A_baseline": {"outdir": BASELINE_OUTDIR, "error_rate_json": baseline_json,
                        "snprc_json": P.RESULTS_DIR / "simval_snp_refcall" / f"{BASELINE_ROW_KEY}.json"},
        "B_noerror": {"outdir": arm_b_dir, "error_rate_json": None,
                       "snprc_json": Path(args.arm_b_snprc_json) if args.arm_b_snprc_json else None},
        "C_redraw": {"outdir": arm_c_dir, "error_rate_json": None,
                      "snprc_json": Path(args.arm_c_snprc_json) if args.arm_c_snprc_json else None},
    }

    metrics_rows = []
    for arm_name, cfg_ in arms.items():
        m = load_arm_metrics(cfg_["outdir"], arm_name)
        if cfg_["snprc_json"] and cfg_["snprc_json"].exists():
            m.update(load_snprc_metrics(cfg_["snprc_json"]))
        else:
            m.update({"snprc_error_rate": None, "snprc_partial_error_rate": None,
                      "snprc_compared_sites": None})
        m["arm"] = arm_name
        metrics_rows.append(m)

    # --- PS4G aggregate background ---
    ps4g_rows = []
    for arm_name, cfg_ in arms.items():
        ps4g_path = cfg_["outdir"] / "raw.ps4g"
        if ps4g_path.exists():
            s = ps4g_background_summary(ps4g_path, true_parents)
            s["arm"] = arm_name
            ps4g_rows.append(s)

    # --- L3 confusion matrix, arms B/C only ---
    confusion = {}
    for arm_name in ("B_noerror", "C_redraw"):
        raw_tsv = arms[arm_name]["outdir"] / "raw.tsv"
        if not raw_tsv.exists():
            continue
        matrix, status_counts, per_source_totals, unlabeled = build_confusion_matrix(raw_tsv)
        summary = summarize_confusion(matrix, per_source_totals, true_parents)
        confusion[arm_name] = {
            "matrix": matrix, "status_counts": status_counts,
            "per_source_totals": per_source_totals, "unlabeled_reads": unlabeled,
            "summary": summary,
        }

    # --- write outputs ---
    import csv
    with open(args.out_tsv, "w", newline="") as f:
        cols = ["arm", "error_rate", "partial_error_rate", "compared_sites",
                "snprc_error_rate", "snprc_partial_error_rate", "snprc_compared_sites",
                "align_dropped_idx", "align_het_scale", "imputed_records"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", delimiter="\t")
        w.writeheader()
        for row in metrics_rows:
            w.writerow(row)

    md_lines = ["# Zero-sequencing-error control: IDX-HYB / Oh43xIl14H / 0.1x\n"]
    md_lines.append("## All-sites and SNP+RefCall error rate by arm\n")
    md_lines.append("| arm | error_rate | partial_error_rate | snprc_error_rate | compared_sites |")
    md_lines.append("|---|---|---|---|---|")
    for row in metrics_rows:
        snprc_str = f"{row['snprc_error_rate']:.4%}" if row['snprc_error_rate'] is not None else "N/A"
        md_lines.append(
            f"| {row['arm']} | {row['error_rate']:.4%} | {row['partial_error_rate']:.4%} | "
            f"{snprc_str} | {row['compared_sites']} |"
        )

    md_lines.append("\n## PS4G aggregate background\n")
    md_lines.append("| arm | total_unique_counts | mean_true_parent_count | "
                     "median_nonparent_count | background_ratio |")
    md_lines.append("|---|---|---|---|---|")
    for row in ps4g_rows:
        md_lines.append(
            f"| {row['arm']} | {row['total_unique_counts']} | "
            f"{row['mean_true_parent_count']:.1f} | {row['median_nonparent_count']:.1f} | "
            f"{row['background_ratio']:.3f} |"
        )

    for arm_name, c in confusion.items():
        md_lines.append(f"\n## L3 true-source x credited-founder confusion, {arm_name}\n")
        md_lines.append(f"status counts (all reads, labeled+unlabeled): {c['status_counts']}\n")
        md_lines.append("| true source | n_reads | self_credit_rate (of PLACED) | "
                         "mean_other_founder_credit | background_ratio |")
        md_lines.append("|---|---|---|---|---|")
        for source, s in c["summary"].items():
            md_lines.append(
                f"| {source} | {s['n_reads_from_source']} | {s['self_credit_rate']:.3%} | "
                f"{s['mean_other_founder_credit']:.1f} | {s['background_ratio']:.3f} |"
            )

    Path(args.out_md).write_text("\n".join(md_lines) + "\n")
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    full_json = Path(args.out_tsv).with_suffix(".full.json")
    full_json.write_text(json.dumps({
        "metrics": metrics_rows, "ps4g": ps4g_rows, "confusion": confusion,
    }, indent=1, default=str))

    print(f"Wrote {args.out_tsv}, {args.out_md}, {full_json}")
    for row in metrics_rows:
        print(row)


if __name__ == "__main__":
    main()
