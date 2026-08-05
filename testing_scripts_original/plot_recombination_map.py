#!/usr/bin/env python3
"""
Plot maize genetic recombination rate by chromosome position.

Two datasets are overlaid per chromosome panel:
  1. Published bin map (12915_2015_187_MOESM2_ESM.csv)
       rate = cM / bin_size_Mb, averaged across all families
  2. Minimac4 cleaned map (combined_cleaned.map)
       rate = delta_cM / delta_pos_Mb (local derivative, smoothed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ───────────────────────────────────────────────────────────────────
CSV_MAP   = "/workdir/irk9/data/maps/maize/12915_2015_187_MOESM2_ESM.csv"
CLEAN_MAP = "/workdir/irk9/data/maps/maize/minimac_impute_maize_map/combined_cleaned.map"
OUT_FILE  = "/local/workdir/irk9/data/phg-maize/bellas_scripts/recombination_map.png"

# ── options ─────────────────────────────────────────────────────────────────
SHOW_FAMILIES  = False   # True → thin line per family behind the mean
SMOOTH_WINDOW  = 15      # rolling window (# SNPs) for cleaned-map derivative
RATE_YLIM      = (0, 20) # cap y-axis (cM/Mb); set None for auto (reveals outlier spikes)
MIN_BIN_MB     = 0.05    # drop CSV bins smaller than this (Mb) — tiny bins create artifactual spikes
CHR_ORDER      = [f"chr{i}" for i in range(1, 11)]
GRID_ROWS, GRID_COLS = 2, 5

# ── palette (validated categorical) ─────────────────────────────────────────
C_BIN_MEAN  = "#2a78d6"   # blue  – bin map mean across families
C_CLEAN     = "#eb6834"   # orange – cleaned minimac4 map
C_BIN_FAM   = "#aac8f0"   # faint blue – per-family lines (if SHOW_FAMILIES)

# ── load CSV bin map ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV_MAP, skipinitialspace=True)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Chr":              "chr",
    "Start":            "start_mb",
    "Stop":             "stop_mb",
    "Bin size (Mb)":    "bin_mb",
    "Genetic map (cM)": "cM",
    "Family":           "family",
})
df["chr"] = "chr" + df["chr"].astype(str)
df["midpoint_mb"] = (df["start_mb"] + df["stop_mb"]) / 2
df = df[df["bin_mb"] >= MIN_BIN_MB]  # drop zero-size and very small bins
# "Genetic map (cM)" is cumulative → compute per-bin delta first
df = df.sort_values(["chr", "family", "midpoint_mb"])
df["delta_cM"] = df.groupby(["chr", "family"])["cM"].diff()
df["rate"] = df["delta_cM"] / df["bin_mb"]   # cM / Mb
# to use raw cumulative cM / binSize instead: df["rate"] = df["cM"] / df["bin_mb"]
df = df[df["rate"] >= 0]           # drop first-bin NaN rows and any negatives

# mean across families per chr/bin midpoint
mean_df = (
    df.groupby(["chr", "Marker", "midpoint_mb"])["rate"]
    .mean()
    .reset_index()
    .sort_values(["chr", "midpoint_mb"])
)

# ── load cleaned map ─────────────────────────────────────────────────────────
cmap = pd.read_csv(
    CLEAN_MAP, sep=r"\s+", header=None,
    names=["pos", "chr", "cM"]
)
cmap["pos_mb"] = cmap["pos"] / 1e6
cmap = cmap.sort_values(["chr", "pos_mb"]).reset_index(drop=True)

def local_rate(grp, window=SMOOTH_WINDOW):
    """Compute local recombination rate as delta_cM / delta_pos_Mb."""
    g = grp.sort_values("pos_mb").copy()
    g["rate"] = g["cM"].diff() / g["pos_mb"].diff()
    g["rate"] = (
        g["rate"]
        .clip(lower=0)
        .rolling(window, center=True, min_periods=1)
        .mean()
    )
    return g

cmap = cmap.groupby("chr", group_keys=False).apply(local_rate)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    GRID_ROWS, GRID_COLS,
    figsize=(18, 7),
    sharey=False,
    constrained_layout=True,
)

for ax, ch in zip(axes.flatten(), CHR_ORDER):
    sub_csv  = mean_df[mean_df["chr"] == ch]
    sub_cmap = cmap[cmap["chr"] == ch]

    if SHOW_FAMILIES:
        for _, fgrp in df[df["chr"] == ch].groupby("family"):
            fgrp = fgrp.sort_values("midpoint_mb")
            ax.plot(fgrp["midpoint_mb"], fgrp["rate"],
                    color=C_BIN_FAM, lw=0.4, alpha=0.4)

    ax.plot(sub_csv["midpoint_mb"], sub_csv["rate"],
            color=C_BIN_MEAN, lw=1.4, label="Bin map (mean)")
    ax.plot(sub_cmap["pos_mb"], sub_cmap["rate"],
            color=C_CLEAN, lw=1.0, alpha=0.85, label="Cleaned map")

    if RATE_YLIM:
        ax.set_ylim(RATE_YLIM)

    ax.set_title(ch, fontsize=9, fontweight="bold")
    ax.set_xlabel("Position (Mb)", fontsize=7)
    ax.set_ylabel("cM / Mb", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.5, zorder=0)

axes.flatten()[0].legend(fontsize=7, frameon=False)

fig.suptitle("Maize recombination rate by chromosome", fontsize=12, fontweight="bold")
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.show()
