#!/usr/bin/env python
"""
Does B73/CML103 read-support co-occurrence (the identity-by-descent (IBD)
signature quantified genome-wide by bp_distance_decay.py) specifically
concentrate at the loci where B73xCML103 actually decodes wrong, or is it
just present everywhere at a similar background rate regardless of
whether an error occurs there? bp_distance_decay.py showed the genome-wide
AVERAGE level of persistence is similar across pairs regardless of
whether that pair shows founder-specific anchor dropout (B73xOh43) or
mutual reference confusion (B73xCML103) -- so it doesn't discriminate
MECHANISM. This asks a different, more local question: WITHIN B73xCML103,
are the specific loci that decode wrong the ones with locally elevated
B73+CML103 co-support, the way item 4 tested anchor density at error loci
(results/ril2_error_regions/anchor_density_vs_error.json)?

Reuses extract_error_regions.py's true-founder-labeled wrong intervals and
the same width-matched background sampling used throughout this line of
work; local co-support rate = mean(B73 column nonzero AND CML103 column
nonzero) over the raw.npy rows whose position falls inside the region.
"""
import argparse
import bisect
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_paths as P  # noqa: E402
from extract_error_regions import extract_wrong_intervals  # noqa: E402
from simval_oracle_bed import build_ril_mosaics  # noqa: E402

SOURCE_K = 25
FAI = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai"


def load_chrom_lengths():
    lengths = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            if name.startswith("chr") and name[3:].isdigit():
                lengths[name] = int(length)
    return lengths


def sample_background(widths, chrom_lengths, n, seed):
    chroms = list(chrom_lengths.keys())
    total_len = sum(chrom_lengths.values())
    weights = [chrom_lengths[c] / total_len for c in chroms]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chrom = rng.choices(chroms, weights=weights, k=1)[0]
        width = rng.choice(widths)
        s = rng.randint(0, max(0, chrom_lengths[chrom] - width))
        out.append({"chrom": chrom, "start": s, "end": s + width})
    return out


def build_position_index(bins_df, ambiguous):
    """Per-chrom sorted position array + parallel ambiguous-indicator array."""
    idx = {}
    for chrom, rows in bins_df.groupby("contig", sort=False).indices.items():
        rows = np.sort(rows)
        idx[chrom] = (bins_df.loc[rows, "bin"].to_numpy(dtype=np.int64) * 256, ambiguous[rows])
    return idx


def local_rate(pos_idx, chrom, start, end):
    if chrom not in pos_idx:
        return None
    positions, amb = pos_idx[chrom]
    lo = bisect.bisect_left(positions, start)
    hi = bisect.bisect_left(positions, end)
    if hi <= lo:
        return None
    return float(amb[lo:hi].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individual", default="B73xCML103")
    ap.add_argument("--parent-a", default="B73")
    ap.add_argument("--parent-b", default="CML103")
    ap.add_argument("--coverage", default="0.5")
    ap.add_argument("--tag", default="unfiltered-bin")
    ap.add_argument("--n-background", type=int, default=1000)
    cli = ap.parse_args()

    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{cli.individual}__{cli.coverage}x__{cli.tag}"
    gamete_names = hae.load_gamete_names(outdir / "raw.npy.gametes.tsv")
    a_idx = gamete_names.index(cli.parent_a)
    b_idx = gamete_names.index(cli.parent_b)
    arr = np.load(outdir / "raw.npy", mmap_mode="r")
    feats = np.asarray(arr[:, :SOURCE_K])
    ambiguous = ((feats[:, a_idx] != 0) & (feats[:, b_idx] != 0)).astype(np.float64)
    base_rate = ambiguous.mean()
    print(f"{cli.individual} {cli.coverage}x: genome-wide base co-support rate = {base_rate:.4f}")

    # fanout: how many of the 25 founders (not just A/B) have nonzero support per row --
    # narrow (~2) means the ambiguity at that row is specifically A vs B; higher means
    # broader multi-founder confusion, a qualitatively different kind of ambiguity.
    fanout = (feats != 0).sum(axis=1).astype(np.float64)

    bins_df = pd.read_csv(outdir / "raw.npy.bins.tsv", sep="\t")
    pos_idx = build_position_index(bins_df, ambiguous)
    fanout_idx = build_position_index(bins_df, fanout)

    mosaic_h1, _, label_to_name = build_ril_mosaics("IDX-RIL2", cli.parent_a, cli.parent_b)
    bed_dir = outdir / "bed"
    sample = f"{cli.individual}_ril2_{cli.tag}"
    wrong, wrong_bp, total_bp = extract_wrong_intervals(bed_dir, sample, mosaic_h1, label_to_name)
    print(f"{len(wrong)} wrong intervals, wrong_bp={wrong_bp:,} total_bp={total_bp:,}")

    chrom_lengths = load_chrom_lengths()
    widths = [max(1, w["width"]) for w in wrong]
    bg = sample_background(widths, chrom_lengths, cli.n_background, seed=42)

    err_rates = [r for w in wrong if (r := local_rate(pos_idx, w["chrom"], w["start"], w["end"])) is not None]
    bg_rates = [r for w in bg if (r := local_rate(pos_idx, w["chrom"], w["start"], w["end"])) is not None]
    print(f"scored: error n={len(err_rates)}  background n={len(bg_rates)}")

    print(f"error region local co-support:      median={np.median(err_rates):.4f}  mean={np.mean(err_rates):.4f}")
    print(f"background region local co-support: median={np.median(bg_rates):.4f}  mean={np.mean(bg_rates):.4f}")
    print(f"genome-wide base rate:               {base_rate:.4f}")

    u, p = sps.mannwhitneyu(err_rates, bg_rates, alternative="two-sided")
    print(f"Mann-Whitney error vs background: p={p:.4g}")
    ratio = np.median(err_rates) / np.median(bg_rates) if np.median(bg_rates) > 0 else float("nan")
    print(f"error/background median ratio: {ratio:.2f}x")

    err_fanout = [r for w in wrong if (r := local_rate(fanout_idx, w["chrom"], w["start"], w["end"])) is not None]
    bg_fanout = [r for w in bg if (r := local_rate(fanout_idx, w["chrom"], w["start"], w["end"])) is not None]
    print(f"\nerror region local fanout (mean # founders w/ nonzero support): "
          f"median={np.median(err_fanout):.2f}  mean={np.mean(err_fanout):.2f}")
    print(f"background region local fanout:                                 "
          f"median={np.median(bg_fanout):.2f}  mean={np.mean(bg_fanout):.2f}")
    uf, pf = sps.mannwhitneyu(err_fanout, bg_fanout, alternative="two-sided")
    print(f"Mann-Whitney fanout error vs background: p={pf:.4g}")


if __name__ == "__main__":
    main()
