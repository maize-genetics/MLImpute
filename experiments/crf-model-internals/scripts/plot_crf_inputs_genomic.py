#!/usr/bin/env python
"""
Plot what actually feeds the CRF (excluding emission scores) across the
genome: gate g and recombination cost c, for Oh43/Il14H (inbred) and
Oh43xIl14H (hybrid), same footing as the encoder-track plots.

Standalone: numpy + matplotlib only.

Usage:
    python plot_crf_inputs_genomic.py --indir results/crf_inputs_genomic --outdir out
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [
    ("IDX-INBRED__Oh43__0.1x.npz", "#2c6fbb"),
    ("IDX-INBRED__Il14H__0.1x.npz", "#2e8b57"),
    ("IDX-HYB__Oh43xIl14H__0.1x.npz", "#c0392b"),
]


def rolling_mean(y, order, win=51):
    s = np.convolve(y[order], np.ones(win) / win, mode="same")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/crf_inputs_genomic")
    ap.add_argument("--outdir", default="crf_inputs_genomic_figs")
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for fname, color in ROWS:
        p = indir / fname
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        data[fname] = dict(bp=d["bp"], gate=d["gate"], recomb_cost=d["recomb_cost"],
                            stay_bonus=float(d["stay_bonus"]), label=str(d["label"]), color=color)
    if not data:
        raise SystemExit(f"No .npz in {indir}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for fname, d in data.items():
        bp_mb = d["bp"] / 1e6
        order = np.argsort(bp_mb)
        axes[0].plot(bp_mb, d["gate"], ".", ms=2, alpha=0.4, color=d["color"])
        axes[0].plot(bp_mb[order], rolling_mean(d["gate"], order), "-", lw=1.8,
                     color=d["color"], label=d["label"])
        axes[1].plot(bp_mb, d["recomb_cost"], ".", ms=2, alpha=0.4, color=d["color"])
        axes[1].plot(bp_mb[order], rolling_mean(d["recomb_cost"], order), "-", lw=1.8,
                     color=d["color"], label=d["label"])
    axes[0].set_ylabel("gate g\n(sigmoid, 0-1)")
    axes[0].set_title("Gate g across the genome (modulates emission inside the encoder; "
                       "not passed to the CRF)")
    axes[0].legend(fontsize=8, markerscale=4)
    axes[1].set_ylabel("recomb cost c\n(softplus, switch-discouraging)")
    axes[1].set_xlabel("chr1 position (Mb)")
    sb = next(iter(data.values()))["stay_bonus"]
    axes[1].set_title(f"Recombination/switch cost c across the genome -- THE per-site signal "
                       f"the CRF actually uses (plus global stay_bonus={sb:.3f})")
    axes[1].legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(outdir / "fig01_gate_recomb_genomic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'fig01_gate_recomb_genomic.png'}")

    # Quick summary stats
    print("\n=== summary ===")
    for fname, d in data.items():
        print(f"{d['label']:22s} gate: mean={d['gate'].mean():.4f} std={d['gate'].std():.4f} "
              f"min={d['gate'].min():.4f}  |  recomb_cost: mean={d['recomb_cost'].mean():.3f} "
              f"std={d['recomb_cost'].std():.3f}")


if __name__ == "__main__":
    main()
