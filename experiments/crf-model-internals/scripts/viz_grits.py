#!/usr/bin/env python
"""
Standalone visualizer for the train-vs-real support profile and model-
internals dump (support_profile.py / dump_model_internals.py). numpy +
matplotlib ONLY -- no torch, no repo imports, no seaborn (installed nowhere
in this project's envs). Meant to run on your laptop after scp'ing down
`results/support_profile*` and `results/model_internals/`.

Usage:
    python viz_grits.py --indir results --outdir grits_viz_out

Expected --indir layout (matches what's produced on the cluster):
    support_profile.tsv
    support_profile_hist.json
    model_internals/manifest.json
    model_internals/<row_id>.npz  (one per row dumped)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_COLORS = {"inbred": "#2c6fbb", "hybrid": "#c0392b", "ril": "#e08a1e"}
SIM_COLOR = "#888888"


def savefig(fig, outdir, name):
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path.name


def load_support(indir):
    tsv = indir / "support_profile.tsv"
    hist = indir / "support_profile_hist.json"
    import csv
    rows = list(csv.DictReader(open(tsv), delimiter="\t")) if tsv.exists() else []
    h = json.load(open(hist)) if hist.exists() else {}
    return rows, h


def load_internals(indir):
    mdir = indir / "model_internals"
    manifest = json.load(open(mdir / "manifest.json")) if (mdir / "manifest.json").exists() else None
    npz = {}
    if mdir.exists():
        for p in sorted(mdir.glob("*.npz")):
            npz[p.stem] = np.load(p)
    return manifest, npz


# --------------------------------------------------------------------------- #

def fig01_support_by_k(rows, outdir):
    sim = [r for r in rows if r["source"] == "sim"]
    real = [r for r in rows if r["source"] == "real"]
    if not sim:
        return None
    by_k = {}
    for r in sim:
        k = int(r["k"])
        by_k.setdefault(k, []).append(r)
    ks = sorted(by_k)
    carried = [np.mean([float(r["credit_carried"]) for r in by_k[k]]) for k in ks]
    bg = [np.mean([float(r["credit_background"]) for r in by_k[k] if r["credit_background"] not in (None, "")]) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, carried, "o-", color="#2c6fbb", label="sim: carried founder")
    ax.plot(ks, bg, "o-", color=SIM_COLOR, label="sim: background founder")
    for r in real:
        k = int(r["k"])
        color = REAL_COLORS.get(r["kind"], "black")
        ax.scatter([k], [float(r["credit_carried"])], marker="*", s=220, color=color,
                   edgecolor="black", zorder=5,
                   label=f"real {r['kind']} carried ({r['row_id']})")
        ax.scatter([k], [float(r["credit_background"])], marker="v", s=100, color=color,
                   edgecolor="black", zorder=5)
    ax.set_xlabel("k (founders carried per individual)")
    ax.set_ylabel("genome-wide credit rate")
    ax.set_title("Training support margin vs founder count k\n"
                  "(stars/triangles = real carried/background)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    return savefig(fig, outdir, "fig01_support_by_k")


def fig02_cardinality(hist, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sim_card = hist.get("sim", {}).get("cardinality", {})
    if sim_card:
        xs = sorted(int(k) for k in sim_card)
        tot = sum(sim_card.values())
        ys = [sim_card[str(x)] / tot for x in xs]
        ax.plot(xs, ys, color=SIM_COLOR, label="sim (all k pooled)", lw=2)
    for row_id, h in hist.get("real", {}).items():
        card = h.get("cardinality", {})
        if not card:
            continue
        xs = sorted(int(k) for k in card)
        tot = sum(card.values())
        ys = [card[str(x)] / tot for x in xs]
        ax.plot(xs, ys, label=row_id, alpha=0.8)
    ax.set_xlabel("founders matched at a site (gameteSet cardinality, of 24)")
    ax.set_ylabel("fraction of sites")
    ax.set_title("Per-site founder-match cardinality: sim vs real")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return savefig(fig, outdir, "fig02_cardinality")


def fig03_feature_values(hist, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sim_fv = hist.get("sim", {}).get("feature_values", {})
    if sim_fv:
        xs = sorted(int(k) for k in sim_fv)
        tot = sum(sim_fv.values())
        ys = [sim_fv[str(x)] / tot for x in xs]
        ax.bar([x - 0.15 for x in xs], ys, width=0.3, color=SIM_COLOR, label="sim")
    for i, (row_id, h) in enumerate(hist.get("real", {}).items()):
        fv = h.get("feature_values", {})
        if not fv:
            continue
        xs = sorted(int(k) for k in fv)
        tot = sum(fv.values())
        ys = [fv[str(x)] / tot for x in xs]
        ax.bar([x + 0.15 + 0.05 * i for x in xs], ys, width=0.1, alpha=0.8, label=row_id)
    ax.set_yscale("log")
    ax.set_xlabel("feature cell value (0 = no support, 1 = one read, 2+ = multiple reads)")
    ax.set_ylabel("fraction of cells (log scale)")
    ax.set_title("Feature-value histogram: sim is strictly binary, real has counts")
    ax.set_xlim(-0.5, 8.5)
    ax.legend(fontsize=7)
    return savefig(fig, outdir, "fig03_feature_values")


def fig04_affinity(npz, manifest, outdir):
    rows_meta = {r["row_id"]: r for r in manifest["rows"]} if manifest else {}
    real_rows = [rid for rid in npz if rows_meta.get(rid, {}).get("source") == "real"]
    if not real_rows:
        return None
    fig, axes = plt.subplots(len(real_rows), 1, figsize=(9, 2.6 * len(real_rows)), squeeze=False)
    for i, rid in enumerate(real_rows):
        ax = axes[i, 0]
        d = npz[rid]
        ext = d["ext_emb"][:, 0]  # raw genome-wide credit rate per founder
        names = list(rows_meta[rid]["founder_names"])
        parents = set(rows_meta[rid].get("parents") or [])
        colors = ["#c0392b" if n in parents else "#888888" for n in names]
        ax.bar(range(len(names)), ext, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_title(f"{rid}  (red = true parent)")
        ax.set_ylabel("ext_emb\n(raw credit)")
    fig.tight_layout()
    return savefig(fig, outdir, "fig04_affinity")


def fig05_path_structure(rows, manifest, outdir):
    sim = [r for r in rows if r["source"] == "sim"]
    by_k = {}
    for r in sim:
        k = int(r["k"])
        by_k.setdefault(k, []).append(float(r["het_frac"]))
    ks = sorted(by_k)
    het = [np.mean(by_k[k]) for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ks, het, "o-", color=SIM_COLOR, label="sim het_frac by k")
    axes[0].axhline(0.0, color=REAL_COLORS["inbred"], ls="--", label="real inbred (0.0)")
    axes[0].axhline(1.0, color=REAL_COLORS["hybrid"], ls="--", label="real hybrid (1.0)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("het_frac"); axes[0].legend(fontsize=7)
    axes[0].set_title("Heterozygosity fraction: training never reaches 0 or 1")
    axes[0].grid(alpha=0.3)

    if manifest:
        names = [r["row_id"] for r in manifest["rows"]]
        dec = [r["mean_decoded_switches"] for r in manifest["rows"]]
        tru = [r["mean_true_switches"] for r in manifest["rows"]]
        x = np.arange(len(names))
        axes[1].bar(x - 0.2, dec, width=0.4, label="decoded switches/window")
        axes[1].bar(x + 0.2, tru, width=0.4, label="TRUE switches/window")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axes[1].set_title("H2: decoded vs true switch rate per window")
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, outdir, "fig05_path_structure")


def fig06_window_detail(npz, manifest, outdir, window=0):
    rows_meta = {r["row_id"]: r for r in manifest["rows"]} if manifest else {}
    real_rows = [rid for rid in npz if rows_meta.get(rid, {}).get("source") == "real"]
    outputs = []
    for rid in real_rows:
        d = npz[rid]
        X = d["X"][window]                      # [T,24]
        fm = d["founder_marginal"][window]       # [T,24]
        names = list(rows_meta[rid]["founder_names"])
        parents = rows_meta[rid].get("parents") or []
        p_idx = [names.index(p) for p in parents if p in names]

        fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 3, 1, 1, 1]})
        axes[0].imshow(X.T, aspect="auto", cmap="Greys", interpolation="nearest")
        for pi in p_idx:
            axes[0].axhline(pi, color="red", lw=0.6, alpha=0.5)
        axes[0].set_ylabel("founder"); axes[0].set_title(f"{rid} window {window} -- read support (rows=true parents in red)")

        axes[1].imshow(fm.T, aspect="auto", cmap="viridis", interpolation="nearest")
        for pi in p_idx:
            axes[1].axhline(pi, color="red", lw=0.6, alpha=0.5)
        axes[1].set_ylabel("founder"); axes[1].set_title("per-founder posterior (expected chromosome count)")

        axes[2].plot(d["gate"][window], color="#2c6fbb"); axes[2].set_ylabel("gate g")
        axes[3].plot(d["recomb_cost"][window], color="#e08a1e"); axes[3].set_ylabel("recomb c")
        axes[4].plot(d["pair_acc_site"][window], color="#2e8b57"); axes[4].set_ylabel("site correct")
        axes[4].set_xlabel("site (0..511)")
        fig.tight_layout()
        outputs.append(savefig(fig, outdir, f"fig06_window_detail_{rid}"))
    return outputs


def fig07_gates_recomb(npz, manifest, outdir):
    rows_meta = {r["row_id"]: r for r in manifest["rows"]} if manifest else {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for rid, d in npz.items():
        kind = rows_meta.get(rid, {}).get("kind", rid)
        color = REAL_COLORS.get(kind, None)
        axes[0].hist(d["gate"].ravel(), bins=40, histtype="step", density=True,
                     label=f"{rid} ({kind})", color=color)
        axes[1].hist(d["recomb_cost"].ravel(), bins=40, histtype="step", density=True,
                     label=f"{rid} ({kind})", color=color)
    axes[0].set_xlabel("gate g"); axes[0].set_title("Gate distribution")
    axes[1].set_xlabel("recomb cost c"); axes[1].set_title("Recombination-cost distribution")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    return savefig(fig, outdir, "fig07_gates_recomb")


def fig08_encoder_pca(npz, manifest, outdir):
    rows_meta = {r["row_id"]: r for r in manifest["rows"]} if manifest else {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for rid, d in npz.items():
        kind = rows_meta.get(rid, {}).get("kind", rid)
        color = REAL_COLORS.get(kind, None)
        pca = d["H_pca"].reshape(-1, d["H_pca"].shape[-1])
        idx = np.random.default_rng(0).choice(len(pca), size=min(1500, len(pca)), replace=False)
        ax.scatter(pca[idx, 0], pca[idx, 1], s=4, alpha=0.35, color=color, label=f"{rid} ({kind})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Encoder hidden-state H, first two PCs (per-row local basis)")
    ax.legend(fontsize=6, markerscale=3)
    return savefig(fig, outdir, "fig08_encoder_pca")


def fig09_het_prior(npz, manifest, outdir):
    rows_meta = {r["row_id"]: r for r in manifest["rows"]} if manifest else {}
    names, mass_pen, mass_raw = [], [], []
    het_pen, het_nopen = [], []
    for r in (manifest["rows"] if manifest else []):
        rid = r["row_id"]
        if rid not in npz:
            continue
        d = npz[rid]
        names.append(f"{rid}\n(homo_scale={r['homo_scale_used']})")
        mass_pen.append(float(d["homo_mass_pen"].mean()))
        mass_raw.append(float(d["homo_mass_raw"].mean()))
        het_pen.append(r["het_frac_decoded_pen"])
        het_nopen.append(r["het_frac_decoded_nopen"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, mass_pen, width=0.4, label="with homo penalty")
    axes[0].bar(x + 0.2, mass_raw, width=0.4, label="penalty removed")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("mean posterior mass on ANY homozygous state")
    axes[0].set_title("H3: homozygous posterior mass, with vs without the fixed penalty")
    axes[0].legend(fontsize=7)

    axes[1].bar(x - 0.2, het_pen, width=0.4, label="with penalty (actual decode)")
    axes[1].bar(x + 0.2, het_nopen, width=0.4, label="penalty removed")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("decoded het fraction")
    axes[1].set_title("Decoded heterozygosity: penalty on vs off")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, outdir, "fig09_het_prior")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--outdir", default="grits_viz_out")
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, hist = load_support(indir)
    manifest, npz = load_internals(indir)

    written = []
    print("Generating figures...")
    if rows:
        written.append(fig01_support_by_k(rows, outdir))
    if hist:
        written.append(fig02_cardinality(hist, outdir))
        written.append(fig03_feature_values(hist, outdir))
    if npz and manifest:
        written.append(fig04_affinity(npz, manifest, outdir))
    if rows:
        written.append(fig05_path_structure(rows, manifest, outdir))
    if npz and manifest:
        written.extend(fig06_window_detail(npz, manifest, outdir))
        written.append(fig07_gates_recomb(npz, manifest, outdir))
        written.append(fig08_encoder_pca(npz, manifest, outdir))
        written.append(fig09_het_prior(npz, manifest, outdir))
    written = [w for w in written if w]

    html = ["<!doctype html><html><head><meta charset='utf-8'><title>GRITS diagnostics</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee;} "
            "img{max-width:100%;display:block;margin:1em 0;border:1px solid #444}</style></head><body>",
            "<h1>GRITS train-vs-real support + model internals</h1>"]
    for name in written:
        html.append(f"<h3>{name}</h3><img src='{name}'>")
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html))
    print(f"\nWrote {len(written)} figures + {outdir/'index.html'}")


if __name__ == "__main__":
    main()
