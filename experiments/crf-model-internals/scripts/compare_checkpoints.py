#!/usr/bin/env python
"""
Multi-way before/after comparison of dump_model_internals.py runs (one per
checkpoint) on the same real rows. Reads each tag's manifest.json (already
written by dump_model_internals.py --ckpt <...>) plus the per-row .npz files
for the RIL homo/het-truth split and the homo_mass_pen mechanism check, and
writes a markdown table with one column per tag.

Usage (Tier 2, 3-way):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/compare_checkpoints.py \
        --tags diploid-affinity-sim512-h3 breedpop-sparse-affinity-learnedhet \
               breedpop-sparse-constant-affinity-learnedhet \
        --out results/tier2_evaluation.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT_ROOT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/model_internals")
METRICS = ["pair_acc", "hap_acc", "marginal_hap_acc",
           "mean_decoded_switches", "mean_true_switches",
           "het_frac_decoded_pen"]


def load_manifest(tag):
    return json.load(open(OUT_ROOT / tag / "manifest.json"))


def homo_mass_pen_mean(tag, row_id):
    p = OUT_ROOT / tag / f"{row_id}.npz"
    if not p.exists():
        return None
    return float(np.load(p)["homo_mass_pen"].mean())


def ril_homo_het_split(tag, row_id="IDX-RIL__Oh43xIl14H__0.1x"):
    p = OUT_ROOT / tag / f"{row_id}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    tl, th = d["true_lo"], d["true_hi"]
    pred_lo, pred_hi = d["pred_lo"], d["pred_hi"]
    correct = ((pred_lo == tl) & (pred_hi == th)).mean(1)
    is_homo = tl[:, 0] == th[:, 0]
    return dict(
        n_homo=int(is_homo.sum()), homo_acc=float(correct[is_homo].mean()) if is_homo.any() else None,
        n_het=int((~is_homo).sum()), het_acc=float(correct[~is_homo].mean()) if (~is_homo).any() else None,
    )


def fmt(v):
    return f"{v:.4f}" if v is not None else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                     help="Checkpoint dump tags, in report order (first is the reference "
                          "baseline for delta columns).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifests = {tag: {r["row_id"]: r for r in load_manifest(tag)["rows"]} for tag in args.tags}
    baseline_tag = args.tags[0]
    common_rows = [r for r in manifests[baseline_tag]
                   if all(r in manifests[t] for t in args.tags)]

    lines = ["# Multi-checkpoint evaluation: real 0.1x rows\n",
             "Tags (in column order): " + ", ".join(f"`{t}`" for t in args.tags) + ".\n",
             "| row | metric | " + " | ".join(args.tags) +
             (" | delta (last - first) |" if len(args.tags) > 1 else " |"),
             "|---|---|" + "---|" * len(args.tags) + ("---|" if len(args.tags) > 1 else "")]
    for row_id in common_rows:
        for metric in METRICS:
            vals = [manifests[t][row_id][metric] for t in args.tags]
            row = f"| {row_id} | {metric} | " + " | ".join(f"{v:.4f}" for v in vals)
            if len(args.tags) > 1:
                row += f" | {vals[-1]-vals[0]:+.4f} |"
            else:
                row += " |"
            lines.append(row)
        hm = [homo_mass_pen_mean(t, row_id) for t in args.tags]
        row = f"| {row_id} | homo_mass_pen (mean) | " + " | ".join(fmt(v) for v in hm)
        if len(args.tags) > 1 and hm[0] is not None and hm[-1] is not None:
            row += f" | {hm[-1]-hm[0]:+.4f} |"
        else:
            row += " |"
        lines.append(row)

    lines.append("\n## RIL homozygous vs heterozygous truth split (the H3-per-locus test)\n")
    lines.append("| checkpoint | homo-truth n | homo_acc | het-truth n | het_acc | gap (homo-het) |")
    lines.append("|---|---|---|---|---|---|")
    for tag in args.tags:
        s = ril_homo_het_split(tag)
        if s is None:
            lines.append(f"| {tag} | (no RIL dump found) | | | | |")
            continue
        gap = (s["homo_acc"] - s["het_acc"]) if (s["homo_acc"] is not None and s["het_acc"] is not None) else None
        lines.append(f"| {tag} | {s['n_homo']} | {fmt(s['homo_acc'])} | {s['n_het']} | "
                     f"{fmt(s['het_acc'])} | {fmt(gap)} |")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
