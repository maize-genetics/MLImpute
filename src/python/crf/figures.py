"""
E9: regenerate all report figures into <workdir>/results/figures/.

Two schematics (simulator, model architecture) drawn with matplotlib patches, plus
result figures whose numbers are taken from docs/RESULTS.md (embedded here as data
dicts with the source experiment named, so the script runs standalone and the
figures always match the leaderboard). Run:

    PYTHONPATH=src .pixi/envs/gpu/bin/python src/python/crf/figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BLUE, GREEN, ORANGE, RED, GREY = "#2c6fbb", "#2e8b57", "#e08a1e", "#c0392b", "#888"


# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1.0, alpha=0.85))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=5)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="black"))


def fig_simulator(out):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("GRITS simulator: founder panel → recombination mosaic → reads → "
                 "match-feature matrix", fontsize=12, weight="bold")
    _box(ax, (0.2, 4.6), 2.5, 1.1,
         "Coalescent / Ewens GEM(θ)\nfounder panel (K=24)\nIBD lineages → shared\nmini-haplotypes", BLUE)
    _box(ax, (0.2, 2.9), 2.5, 1.0,
         "E5: per-individual\nfounder subset (2–24)\n100 windows / individual", GREEN)
    _box(ax, (0.2, 1.2), 2.5, 1.1,
         "E7: per-individual\ninbreeding F\n(spike F=1 + U[0,1])", ORANGE)
    _box(ax, (3.4, 3.4), 2.6, 1.6,
         "Recombination mosaic\nH1, H2 founder paths\n(0–8 crossovers)\nrate map (E2)", "#cfe0f3")
    _box(ax, (6.7, 3.4), 2.5, 1.6,
         "Single-gamete reads\n1 read/site from H1 or H2\nmini-hap of L=16 SNPs\n+ dropout (bad-frac)", "#d6efe0")
    _box(ax, (9.6, 3.4), 2.2, 1.6,
         "Match-feature matrix\n[T × K] 0/1\n(founder f matches read)\n+ labels H1,H2", "#f6e2c8")
    _box(ax, (6.7, 1.0), 2.5, 1.4,
         "Eval-only truth\n• IBD lineage (E-IBD)\n• SNP panel (E6)\n• F per window (E7)", "#eee")
    for a, b in [((2.7, 5.1), (3.4, 4.4)), ((2.7, 3.4), (3.4, 4.0)),
                 ((2.7, 1.8), (3.4, 3.6)), ((6.0, 4.2), (6.7, 4.2)),
                 ((9.2, 4.2), (9.6, 4.2)), ((7.9, 3.4), (7.9, 2.4))]:
        _arrow(ax, a, b)
    ax.text(7.95, 0.5, "het signal (E7): with reads ordered by position, LOW correlation "
            "between adjacent reads ⇒ heterozygous window; HIGH ⇒ homozygous",
            fontsize=8, style="italic", ha="center", color=RED)
    fig.savefig(out / "fig1_simulator.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("FounderPathEncoder + NeuralCRF (shared haploid / diploid)",
                 fontsize=12, weight="bold")
    _box(ax, (0.2, 2.6), 1.8, 1.4, "Match matrix\n[B,T,K]\nlog1p → cell\nembedding", "#f6e2c8")
    _box(ax, (2.4, 2.6), 1.9, 1.4, "Founder attention\npool (per site)\nmasked by\nfounder_mask", BLUE)
    _box(ax, (4.7, 2.6), 1.9, 1.4, "Transformer\nencoder\n(d=256, L=6)\n+ pos-enc", "#cfe0f3")
    _box(ax, (7.0, 4.1), 2.0, 1.1, "per-site emissions\nemis[B,T,K]\n+ gate g", GREEN)
    _box(ax, (7.0, 2.6), 2.0, 1.1, "transition cost\nc_t (recomb head)", "#d6efe0")
    _box(ax, (9.6, 3.2), 2.1, 1.4, "NeuralCRF\nViterbi / NLL\nfounder path\n(stay/switch, no\n[B,T,P,P])", ORANGE)
    _box(ax, (4.7, 0.5), 1.9, 1.1, "E5 affinity →\nemission bias /\nhard cutoff", "#e7d3ef")
    _box(ax, (7.0, 0.5), 2.0, 1.1, "E7 het →\nadaptive homo-\npenalty (diploid)", "#f3d6d6")
    for a, b in [((2.0, 3.3), (2.4, 3.3)), ((4.3, 3.3), (4.7, 3.3)),
                 ((6.6, 3.5), (7.0, 4.5)), ((6.6, 3.1), (7.0, 3.1)),
                 ((9.0, 4.5), (9.6, 4.2)), ((9.0, 3.1), (9.6, 3.6))]:
        _arrow(ax, a, b)
    _arrow(ax, (5.7, 1.6), (7.4, 4.1))      # E5 affinity → emissions
    _arrow(ax, (8.0, 1.6), (8.2, 4.1))      # E7 het → (pair) emissions
    ax.text(6, 0.05, "diploid: pair-state emis_p = emis[i]+emis[j] over P=325 pairs; "
            "same encoder, zero added params", fontsize=8, style="italic", ha="center")
    fig.savefig(out / "fig2_architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- result figures (numbers from docs/RESULTS.md) ------------------------- #
def fig_e1_hmm(out):
    arms = ["HMM\n(Li–Stephens)", "CRF small\n0.88M", "CRF full\n5.07M", "CRF\nno-transition"]
    acc = [0.614, 0.669, 0.682, 0.284]
    cols = [GREY, BLUE, BLUE, RED]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(arms, acc, color=cols)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("held-out founder accuracy")
    ax.set_title("E1-maize: neural CRF vs HMM (real maize)")
    ax.set_ylim(0, 0.8); ax.axhline(0.614, ls="--", color=GREY, lw=0.8)
    fig.savefig(out / "fig3_e1_crf_vs_hmm.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_ibd_ceiling(out):
    bands = ["0 bp", "1-2 bp", "3-5 bp", "6+ bp"]
    crf = [0.9937, 0.9092, 0.8007, 0.7246]
    ceil = [0.9950, 0.9241, 0.8263, 0.7447]
    th = [2, 4, 6, 8, 16]
    ceil_all = [0.7007, 0.7993, 0.8388, 0.8673, 0.9207]
    ceil_hard = [0.5757, 0.6915, 0.7454, 0.7872, 0.8673]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(4)
    a1.bar(x - 0.2, crf, 0.4, label="CRF acc", color=BLUE)
    a1.bar(x + 0.2, ceil, 0.4, label="read-only ceiling", color=ORANGE)
    a1.set_xticks(x); a1.set_xticklabels(bands); a1.set_ylim(0.6, 1.0)
    a1.set_title("E-IBD: CRF sits on the read-only IBD ceiling"); a1.legend(); a1.set_ylabel("accuracy")
    a2.plot(th, ceil_all, "o-", color=BLUE, label="all bands")
    a2.plot(th, ceil_hard, "s-", color=RED, label="6+ bp (hard)")
    a2.set_xscale("log", base=2); a2.set_xticks(th); a2.set_xticklabels(th)
    a2.set_xlabel("θ (relatedness: ←more related   less related→)")
    a2.set_ylabel("read-only ceiling"); a2.set_title("Ceiling rises as relatedness falls (θ-sweep)")
    a2.legend()
    fig.savefig(out / "fig4_ibd_ceiling.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_e5_cutoff(out):
    bands = ["0 bp", "1-2 bp", "3-5 bp", "6+ bp"]
    base = [0.9906, 0.9072, 0.7990, 0.7235]
    cut = [0.9902, 0.9182, 0.8279, 0.7620]
    readc = [0.9955, 0.9229, 0.8228, 0.7439]
    relc = [0.9957, 0.9518, 0.8853, 0.8224]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(4); w = 0.2
    ax.bar(x - 1.5 * w, base, w, label="baseline CRF", color=GREY)
    ax.bar(x - 0.5 * w, cut, w, label="+ hard cutoff (τ=0.17)", color=GREEN)
    ax.bar(x + 0.5 * w, readc, w, label="read-only ceiling", color=ORANGE)
    ax.bar(x + 1.5 * w, relc, w, label="relatedness ceiling", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("E5: hard cutoff beats the read-only ceiling (toward relatedness ceiling)")
    ax.legend(fontsize=8)
    fig.savefig(out / "fig5_e5_cutoff.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_e6_snp(out):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["founder-PATH\naccuracy", "SNP genotype\nconcordance"],
                  [0.8169, 0.9998], color=[BLUE, GREEN])
    for b, v in zip(bars, [0.8169, 0.9998]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy")
    ax.set_title("E6: the IBD path-ceiling is nearly\ninvisible at the SNP level (+0.18)")
    fig.savefig(out / "fig6_e6_snp.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_e7_inbreeding(out, data):
    bands = ["F=1\n(inbred)", "0.66-1", "0.33-0.66", "F<0.33"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(4); w = 0.26
    ax.bar(x - w, data["hp3"], w, label="homo-pen=3 (fixed het prior)", color=ORANGE)
    ax.bar(x, data["hp0"], w, label="homo-pen=0 (no prior)", color=GREY)
    ax.bar(x + w, data["adaptive"], w, label="adaptive (per-individual)", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_ylim(0, 1.0)
    ax.set_ylabel("pair accuracy"); ax.set_xlabel("individual inbreeding F")
    ax.set_title("E7: a fixed homozygosity prior can't serve a mixed-inbreeding panel")
    ax.legend(fontsize=8)
    fig.savefig(out / "fig7_e7_inbreeding.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_e8_speed(out, rows):
    if not rows:
        return
    T = [r[0] for r in rows]; enc = [r[1] for r in rows]
    dec = [r[2] for r in rows]; mem = [r[3] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(T, enc, "o-", label="encoder (Transformer)", color=BLUE)
    a1.plot(T, dec, "s-", label="Viterbi decode", color=GREEN)
    a1.set_xlabel("window length T"); a1.set_ylabel("ms / window"); a1.legend()
    a1.set_title("E8: latency by stage"); a1.set_xscale("log", base=2); a1.set_xticks(T); a1.set_xticklabels(T)
    a2.plot(T, mem, "o-", color=RED)
    a2.set_xlabel("window length T"); a2.set_ylabel("peak GPU memory (MB)")
    a2.set_title("E8: memory scaling"); a2.set_xscale("log", base=2); a2.set_xticks(T); a2.set_xticklabels(T)
    fig.savefig(out / "fig8_e8_speed.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--out", default="docs/figures",
                   help="figure output dir (committed docs artifact, not workdir)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    e7 = dict(hp3=[0.188, 0.237, 0.320, 0.445],     # RESULTS.md E7 (pair_acc by F)
              hp0=[0.818, 0.707, 0.476, 0.171],
              adaptive=[0.757, 0.238, 0.279, 0.388])
    # E8 speed: (T, encoder_ms, viterbi_ms, peak_MB) — RESULTS.md E8, H200 batch=16
    e8 = [(256, 2.71, 7.30, 320), (512, 3.37, 14.81, 586), (1024, 6.24, 29.96, 1117),
          (2048, 12.96, 59.65, 2180), (4096, 29.32, 119.42, 4307)]

    fig_simulator(out)
    fig_architecture(out)
    fig_e1_hmm(out)
    fig_ibd_ceiling(out)
    fig_e5_cutoff(out)
    fig_e6_snp(out)
    fig_e7_inbreeding(out, e7)
    fig_e8_speed(out, e8)
    print(f"figures → {out}")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
