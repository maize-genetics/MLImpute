"""
Allele-sharing / site-frequency-spectrum (SFS) summary across training matrices.

For each (N, T, K+2) matrix (cols 0:K = per-founder read/match counts, col K=H1,
K+1=H2) we count, per site, how many of the K founders carry a nonzero value
("sharing count" = how many founders share the sample's reads/allele). The SFS is
the distribution of that count over sites; the "sharing rate" is its mean / K.

Outputs a 2-panel figure (overlaid SFS + mean-sharing bars) and prints a table.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src \
      .pixi/envs/gpu/bin/python src/python/crf/sfs_sharing.py \
        --out /workdir/esb33/results/sfs_sharing.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, path) — edit here to add files
DEFAULT_FILES = [
    ("Sim inbred",  "/workdir/esb33/data/training/sim_alleles.npy"),
    ("Sim diploid", "/workdir/esb33/data/training/sim_diploid_512.npy"),
    ("Real maize",  "/local/workdir/zrm22/HackathonJun2026/NonCollapsedDatasets/"
                    "singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy"),
    ("Real cassava","/local/workdir/zrm22/HackathonJun2026/NonCollapsedDatasets/"
                    "singleShuffledCassavaDataset/fullMaizeDataset_all_diploid.npy"),
]


def analyze(path, K=24, n_windows=20000, seed=0):
    a = np.load(path, mmap_mode="r")
    N = a.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=min(n_windows, N), replace=False))
    s = np.asarray(a[idx])                              # [n,T,K+2]
    feats = s[:, :, :K]
    h1 = s[:, :, K].astype(int)
    h2 = s[:, :, K + 1].astype(int)

    has = feats != 0                                   # [n,T,K] founder-present
    share = has.sum(axis=2).ravel()                    # founders-with-read per site
    # SFS over k=0..K (fraction of all sites)
    sfs = np.bincount(share, minlength=K + 1)[:K + 1].astype(float)
    sfs /= sfs.sum()
    rate = share.mean() / K                            # mean sharing fraction (background)
    het = float((h1 != h2).mean())
    nz = share > 0

    # True-founder presence: does the labelled founder carry a read at the site?
    n, T = h1.shape
    ii = np.arange(n)[:, None]
    tt = np.arange(T)[None, :]
    h1_has = has[ii, tt, h1]                            # [n,T] bool
    h2_has = has[ii, tt, h2]
    p_h1 = float(h1_has.mean())
    p_h2 = float(h2_has.mean())
    p_either = float((h1_has | h2_has).mean())
    return {
        "sfs": sfs, "rate": rate, "het": het,
        "mean_share": share.mean(), "frac_zero": float((share == 0).mean()),
        "shape": a.shape, "n_sampled": len(idx),
        "median_nz": float(np.median(share[nz])) if nz.any() else 0.0,
        "p_h1": p_h1, "p_h2": p_h2, "p_either": p_either, "background": rate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workdir/esb33/results/sfs_sharing.png")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--n-windows", type=int, default=20000)
    args = p.parse_args()
    K = args.num_parents

    results = []
    print(f"{'file':<14}{'het':>6}{'bg_share':>9}{'H1':>7}{'H2':>7}"
          f"{'either':>8}{'H1/bg':>7}")
    print("-" * 58)
    for label, path in DEFAULT_FILES:
        if not Path(path).exists():
            print(f"{label:<14}  MISSING: {path}")
            continue
        r = analyze(path, K=K, n_windows=args.n_windows)
        results.append((label, r))
        ratio = r["p_h1"] / r["background"] if r["background"] else 0
        print(f"{label:<14}{r['het']:>6.2f}{r['rate']*100:>8.1f}%"
              f"{r['p_h1']*100:>6.0f}%{r['p_h2']*100:>6.0f}%"
              f"{r['p_either']*100:>7.0f}%{ratio:>6.1f}x")

    # ---- figure ----
    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1.5]})

    ks = np.arange(K + 1)
    for (label, r), c in zip(results, colors):
        ax1.plot(ks, r["sfs"] * 100, marker="o", ms=3, lw=1.8, color=c,
                 label=f"{label} (bg {r['rate']*100:.0f}%)")
    ax1.set_xlabel("# founders sharing the site (of 24)")
    ax1.set_ylabel("% of sites")
    ax1.set_title("Site-frequency spectrum (allele sharing)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    # Panel 2: true-founder presence vs background, grouped bars per file.
    labels = [l for l, _ in results]
    nf = len(labels)
    groups = [("background\n(random founder)", "background", "#9ca3af"),
              ("H1", "p_h1", "#2563eb"),
              ("H2", "p_h2", "#16a34a"),
              ("either H1/H2", "p_either", "#dc2626")]
    bw = 0.2
    x = np.arange(nf)
    for gi, (gl, key, c) in enumerate(groups):
        vals = [results[i][1][key] * 100 for i in range(nf)]
        ax2.bar(x + (gi - 1.5) * bw, vals, bw, label=gl, color=c)
    ax2.axhline(95, ls="--", lw=1, color="black", alpha=0.6)
    ax2.text(nf - 0.5, 96, "~95% expected", fontsize=8, ha="right", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("% of sites founder carries a read")
    ax2.set_title("True-founder (H1/H2) presence vs background")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nfigure → {out}")


if __name__ == "__main__":
    main()
