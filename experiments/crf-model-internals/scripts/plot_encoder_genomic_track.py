#!/usr/bin/env python
"""
Plot the encoder hidden state H across genomic position, NOT via PCA:
  1. A raw-H heatmap (dims x genomic position), dimensions sorted by how
     discriminative they are for het/homo (per probe_het_signal.py's
     logistic-regression weights) so real structure shows up in the top
     rows instead of being buried in an arbitrary dimension order.
  2. A single "predicted heterozygosity" track (the same linear probe's
     sigmoid output) along genomic position, for direct visual comparison
     across individuals -- the most interpretable "not a PCA" view.

Standalone: numpy + matplotlib only, run locally on the pulled-down
results/encoder_genomic_track/ directory.

Usage:
    python plot_encoder_genomic_track.py --indir results/encoder_genomic_track --outdir out
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/encoder_genomic_track")
    ap.add_argument("--outdir", default="encoder_genomic_track_figs")
    ap.add_argument("--n-dims-heatmap", type=int, default=64,
                     help="Top-N most discriminative dims to show in the heatmap "
                          "(all 256 is too dense to read).")
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
        data[fname] = dict(H=d["H"], bp=d["bp"], probe_score=d["probe_score"],
                            dim_order=d["dim_order"], label=str(d["label"]), color=color,
                            probe_weights=d["probe_weights"])

    if not data:
        raise SystemExit(f"No .npz files found in {indir}")

    dim_order = next(iter(data.values()))["dim_order"][:args.n_dims_heatmap]

    # --- Figure 0: sample-specific RESIDUAL after removing the shared,
    # genomic-position-locked component (mean H across the 3 individuals at
    # each site) -- isolates real per-sample signal from the ~51% of
    # variance that turned out to be a position confound (mappability/
    # repeat-content/depth shared by all 3 since they align to one
    # reference), which otherwise dominates the raw heatmap visually.
    n_common = min(d["H"].shape[0] for d in data.values())
    Hs_common = {f: d["H"][:n_common].reshape(-1, 256) for f, d in data.items()}
    bp_common = next(iter(data.values()))["bp"][:n_common * data[next(iter(data))]["H"].shape[1]]
    mean_H = np.mean(list(Hs_common.values()), axis=0)
    resid = {f: (Hs_common[f] - mean_H) for f in Hs_common}

    fig, axes = plt.subplots(len(data), 1, figsize=(14, 3.2 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]
    rvmax = max(np.abs(r[:, dim_order]).max() for r in resid.values())
    for ax, fname in zip(axes, data):
        d = data[fname]
        Rf = resid[fname][:, dim_order].T
        bp_mb = bp_common / 1e6
        im = ax.imshow(Rf, aspect="auto", cmap="RdBu_r", vmin=-rvmax, vmax=rvmax,
                        extent=[bp_mb.min(), bp_mb.max(), args.n_dims_heatmap, 0],
                        interpolation="nearest")
        ax.set_ylabel(f"{d['label']}\ndim (same order)")
        ax.set_title(d["label"], fontsize=10)
    axes[-1].set_xlabel("chr1 position (Mb)")
    fig.colorbar(im, ax=axes, label="H minus shared genomic-position mean", shrink=0.6)
    fig.suptitle("Sample-specific RESIDUAL after removing the shared, position-locked\n"
                  "component (mappability/depth confound removed) -- the real zygosity signal", y=1.02)
    fig.savefig(outdir / "fig00_H_residual_heatmap_genomic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'fig00_H_residual_heatmap_genomic.png'}")

    # --- Figure 1: raw-H heatmap, one panel per individual, shared dim order ---
    fig, axes = plt.subplots(len(data), 1, figsize=(14, 3.2 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]
    vmax = max(np.abs(d["H"]).max() for d in data.values())
    for ax, (fname, d) in zip(axes, data.items()):
        Hf = d["H"].reshape(-1, d["H"].shape[-1])[:, dim_order].T  # [n_dims, n_sites]
        bp_mb = d["bp"] / 1e6
        im = ax.imshow(Hf, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[bp_mb.min(), bp_mb.max(), args.n_dims_heatmap, 0],
                        interpolation="nearest")
        ax.set_ylabel(f"{d['label']}\nencoder dim\n(sorted by\nhet-discriminativeness)")
        ax.set_title(d["label"], fontsize=10)
    axes[-1].set_xlabel("chr1 position (Mb)")
    fig.colorbar(im, ax=axes, label="encoder activation", shrink=0.6)
    fig.suptitle(f"Raw encoder output H across the genome (top {args.n_dims_heatmap} "
                 "most het-discriminative dims, NOT PCA)", y=1.0)
    fig.savefig(outdir / "fig01_H_heatmap_genomic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'fig01_H_heatmap_genomic.png'}")

    # --- Figure 2: predicted-heterozygosity probe score along the genome ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for fname, d in data.items():
        bp_mb = d["bp"] / 1e6
        ax.plot(bp_mb, d["probe_score"], ".", ms=2, alpha=0.5, color=d["color"], label=d["label"])
        # smoothed trend (rolling mean over sorted positions) for readability
        order = np.argsort(bp_mb)
        win = 51
        smoothed = np.convolve(d["probe_score"][order], np.ones(win) / win, mode="same")
        ax.plot(bp_mb[order], smoothed, "-", lw=1.8, color=d["color"], alpha=0.9)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="decision boundary")
    ax.set_xlabel("chr1 position (Mb)")
    ax.set_ylabel("linear-probe P(heterozygous)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Encoder-derived \"predicted heterozygosity\" across the genome\n"
                  "(dots = per-site probe output; line = 51-site rolling mean)")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    fig.savefig(outdir / "fig02_probe_score_genomic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'fig02_probe_score_genomic.png'}")

    # --- Figure 3: probe score computed on the RESIDUAL only (confound removed) ---
    w = next(iter(data.values()))["probe_weights"]
    fig, ax = plt.subplots(figsize=(14, 5))
    for fname, d in data.items():
        z = resid[fname] @ w  # mean-centered by construction -- no bias term needed
        bp_mb = bp_common / 1e6
        ax.plot(bp_mb, z, ".", ms=2, alpha=0.5, color=d["color"], label=d["label"])
        order = np.argsort(bp_mb)
        win = 51
        smoothed = np.convolve(z[order], np.ones(win) / win, mode="same")
        ax.plot(bp_mb[order], smoothed, "-", lw=1.8, color=d["color"], alpha=0.9)
    ax.axhline(0.0, color="gray", ls="--", lw=1, label="group mean")
    ax.set_xlabel("chr1 position (Mb)")
    ax.set_ylabel("residual probe logit (H minus shared position-component, projected)")
    ax.set_title("Same probe, but on the SAMPLE-SPECIFIC RESIDUAL only\n"
                  "(shared genomic-position confound removed) -- the real zygosity signal")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    fig.savefig(outdir / "fig03_probe_score_residual_genomic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'fig03_probe_score_residual_genomic.png'}")

    print(f"\nWrote figures to {outdir}")


if __name__ == "__main__":
    main()
