#!/usr/bin/env python
"""
Train-vs-real per-founder read-support profiler.

Computes the SAME statistics on the diploid-affinity training data
(data/training/sim_diploid_512_affinity.npy) and on real simval 0.1x rows
(scratch/simval_eval/<ROW>/windowed_k24_fixdrop23.npy), so the two are
directly comparable rather than eyeballed. See
/home/zrm22/.claude/plans/wondrous-discovering-octopus.md for the
background -- this exists to test (and, on the founder-count axis, refute)
the hypothesis that training's per-founder read support doesn't look like
real alignments.

Per individual/row, this computes:
  - k (number of founders carried)
  - carried vs background genome-wide credit rate -- exactly
    python.crf.train_diploid._founder_affinity's
    r = feats.reshape(-1,K).mean(0), split by the carried mask. Reported as
    BOTH the raw-count mean (what the model's ext_emb actually receives) and
    the binarized (feats>0) mean (comparable to sim, which is already binary,
    and to refbias's PS4G hit_ratio).
  - per-site gameteSet cardinality (feats>0).sum(-1) -- comparable across
    sources
  - per-site read depth feats.sum(-1) -- real only (sim is strictly binary,
    so this is degenerate there)
  - feature-value histogram (0/1/2/...) -- this is where sim's strict
    binary-ness vs real's read counts shows up directly
  - het_frac (h1 != h2, sim only -- real windows carry no h1/h2 truth; see
    dump_model_internals.py for the real-side truth-label path) and switch
    events per window (sim only, same reason)

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/support_profile.py \
        --n-individuals 200 \
        --rows IDX-INBRED__Oh43__0.1x IDX-INBRED__Il14H__0.1x IDX-HYB__Oh43xIl14H__0.1x
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from refbias_parse import split_individual_name, parse_ps4g_header  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIM_DATA = GRITS_WORKDIR / "data/training/sim_diploid_512_affinity.npy"
SIM_IND = GRITS_WORKDIR / "data/training/sim_diploid_512_affinity.ind.npy"
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
RESULTS_DIR = GRITS_WORKDIR / "results"
K_SIM = 24
K_REAL = 24  # windowed_k24_fixdrop23.npy is already trimmed to 24

DEFAULT_ROWS = [
    "IDX-INBRED__Oh43__0.1x",
    "IDX-INBRED__Il14H__0.1x",
    "IDX-HYB__Oh43xIl14H__0.1x",
]


def _row_meta(row_id):
    """'IDX-HYB__Oh43xIl14H__0.1x' -> (dataset_class, kind, individual, coverage)."""
    dataset_class, individual, coverage = row_id.split("__")
    kind = {"INBRED": "inbred", "HYB": "hybrid", "RIL": "ril"}[dataset_class.split("-")[1]]
    return dataset_class.split("-")[0], kind, individual, coverage


# --------------------------------------------------------------------------- #
#  Sim side                                                                    #
# --------------------------------------------------------------------------- #

def profile_sim(n_individuals, seed=0):
    """Stratified-by-k profile over a subsample of training individuals.
    Returns (per_individual DataFrame, hist dict)."""
    a = np.load(SIM_DATA, mmap_mode="r")
    ind = np.load(SIM_IND)
    uniq_ind = np.unique(ind)
    G = int((ind == uniq_ind[0]).sum())
    N = len(uniq_ind)
    rng = np.random.default_rng(seed)
    sel = rng.choice(N, size=min(n_individuals, N), replace=False)
    sel.sort()

    rows = []
    card_hist = defaultdict(int)
    feat_hist = defaultdict(int)
    for i in sel:
        s, e = i * G, (i + 1) * G
        blk = np.asarray(a[s:e])  # [G,512,26]
        f = blk[:, :, :K_SIM].astype(np.float32)
        h1 = blk[:, :, K_SIM].astype(np.int64)
        h2 = blk[:, :, K_SIM + 1].astype(np.int64)
        carried = np.unique(np.concatenate([h1.ravel(), h2.ravel()]))
        carried = carried[(carried >= 0) & (carried < K_SIM)]
        r = f.reshape(-1, K_SIM).mean(0)
        m = np.zeros(K_SIM, dtype=bool)
        m[carried] = True
        card = (f > 0).sum(-1)
        vals, cnts = np.unique(f.astype(np.int64), return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            feat_hist[int(v)] += int(c)
        for v, c in zip(*np.unique(card, return_counts=True)):
            card_hist[int(v)] += int(c)
        switches = ((h1[:, 1:] != h1[:, :-1]) | (h2[:, 1:] != h2[:, :-1])).sum(1)
        rows.append(dict(
            individual=int(i), k=int(len(carried)),
            credit_carried=float(r[m].mean()) if m.any() else np.nan,
            credit_background=float(r[~m].mean()) if (~m).any() else np.nan,
            het_frac=float((h1 != h2).mean()),
            mean_switches_per_window=float(switches.mean()),
            mean_cardinality=float(card.mean()),
        ))
    df = pd.DataFrame(rows)
    hist = {"feature_values": feat_hist, "cardinality": card_hist}
    return df, hist


# --------------------------------------------------------------------------- #
#  Real side                                                                   #
# --------------------------------------------------------------------------- #

def profile_real_row(row_id):
    """Same statistics for one cached simval 0.1x row."""
    row_dir = SIMVAL_EVAL / row_id
    npy_path = row_dir / "windowed_k24_fixdrop23.npy"
    gametes_path = row_dir / "raw.npy.gametes.tsv"
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    dataset_class, kind, individual, coverage = _row_meta(row_id)
    parents = split_individual_name(individual, kind)

    gametes = pd.read_csv(gametes_path, sep="\t").sort_values("gameteIndex")
    source_names = gametes["sampleName"].tolist()  # length 25, pre-drop order
    dropped_txt = row_dir / "windowed_k24_fixdrop23.dropped_idx.txt"
    dropped_idx = int(dropped_txt.read_text()) if dropped_txt.exists() else 23
    kept_names = [n for i, n in enumerate(source_names) if i != dropped_idx]
    name_to_col = {n: i for i, n in enumerate(kept_names)}

    a = np.load(npy_path, mmap_mode="r")
    f = np.asarray(a[:, :, :K_REAL]).astype(np.float32)  # [N,512,24] raw counts
    carried_cols = [name_to_col[p] for p in parents if p in name_to_col]
    m = np.zeros(K_REAL, dtype=bool)
    m[carried_cols] = True

    r_raw = f.reshape(-1, K_REAL).mean(0)
    r_bin = (f > 0).astype(np.float32).reshape(-1, K_REAL).mean(0)
    card = (f > 0).sum(-1)
    depth = f.sum(-1)

    vals, cnts = np.unique(f.astype(np.int64), return_counts=True)
    feat_hist = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}
    cv, cc = np.unique(card, return_counts=True)
    card_hist = {int(v): int(c) for v, c in zip(cv.tolist(), cc.tolist())}

    result = dict(
        row_id=row_id, dataset_class=dataset_class, kind=kind, individual=individual,
        coverage=coverage, parents=parents, k=len(parents),
        credit_raw_carried=float(r_raw[m].mean()) if m.any() else np.nan,
        credit_raw_background=float(r_raw[~m].mean()) if (~m).any() else np.nan,
        credit_bin_carried=float(r_bin[m].mean()) if m.any() else np.nan,
        credit_bin_background=float(r_bin[~m].mean()) if (~m).any() else np.nan,
        mean_cardinality=float(card.mean()),
        mean_depth=float(depth.mean()),
        frac_cells_gt1=float((f > 1).mean()),
        frac_nonzero_cells_gt1=float((f > 1).sum() / max((f > 0).sum(), 1)),
        n_windows=int(f.shape[0]),
    )

    # Cross-check against the PS4G header, when present.
    ps4g_path = row_dir / "raw.ps4g"
    if ps4g_path.exists():
        hdr = parse_ps4g_header(ps4g_path)
        total = hdr["total_unique_counts"]
        gt = hdr["gamete_totals"]
        carried_ps4g = [gt.get(p, 0) / total for p in parents if total]
        bg_names = [n for n in kept_names if n not in parents]
        bg_ps4g = [gt.get(n, 0) / total for n in bg_names if total]
        result["ps4g_credit_carried"] = float(np.mean(carried_ps4g)) if carried_ps4g else np.nan
        result["ps4g_credit_background"] = float(np.mean(bg_ps4g)) if bg_ps4g else np.nan

    return result, {"feature_values": feat_hist, "cardinality": card_hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-individuals", type=int, default=200,
                     help="Sim individuals to subsample (of 1000).")
    ap.add_argument("--rows", nargs="+", default=DEFAULT_ROWS,
                     help="simval 0.1x row ids under scratch/simval_eval/.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Profiling sim training data (n={args.n_individuals} of 1000 individuals)...")
    sim_df, sim_hist = profile_sim(args.n_individuals, args.seed)

    real_results = []
    real_hists = {}
    for row_id in args.rows:
        print(f"Profiling real row {row_id}...")
        res, hist = profile_real_row(row_id)
        real_results.append(res)
        real_hists[row_id] = hist

    # ---- long-form TSV: sim rows + real rows, one row each ----
    long_rows = []
    for _, r in sim_df.iterrows():
        long_rows.append(dict(
            source="sim", row_id=f"sim_ind{int(r.individual):04d}", kind=None, k=int(r.k),
            credit_carried=r.credit_carried, credit_background=r.credit_background,
            mean_cardinality=r.mean_cardinality, het_frac=r.het_frac,
            mean_switches_per_window=r.mean_switches_per_window,
            frac_cells_gt1=0.0,
        ))
    for r in real_results:
        long_rows.append(dict(
            source="real", row_id=r["row_id"], kind=r["kind"], k=r["k"],
            credit_carried=r["credit_bin_carried"], credit_background=r["credit_bin_background"],
            mean_cardinality=r["mean_cardinality"], het_frac=None,
            mean_switches_per_window=None, frac_cells_gt1=r["frac_cells_gt1"],
        ))
    long_df = pd.DataFrame(long_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(RESULTS_DIR / "support_profile.tsv", sep="\t", index=False)

    with open(RESULTS_DIR / "support_profile_hist.json", "w") as fh:
        json.dump({"sim": sim_hist, "real": real_hists}, fh, indent=2)

    # ---- k-stratified sim table (the headline comparison) ----
    strat = sim_df.groupby("k").agg(
        n=("k", "size"),
        credit_carried=("credit_carried", "mean"),
        credit_background=("credit_background", "mean"),
        het_frac=("het_frac", "mean"),
        mean_switches_per_window=("mean_switches_per_window", "mean"),
    ).reset_index()
    strat["margin"] = strat.credit_carried - strat.credit_background

    md = ["# Train-vs-real per-founder read support\n",
          f"Sim: {args.n_individuals} of 1000 training individuals "
          f"(`{SIM_DATA.relative_to(GRITS_WORKDIR)}`), stratified by founder count k. "
          f"Real: {len(args.rows)} cached simval 0.1x rows "
          f"(`scratch/simval_eval/<ROW>/windowed_k24_fixdrop23.npy`).\n",
          "## Sim training data, stratified by k (founders carried per individual)\n",
          "| k | n | carried credit | background credit | margin | het_frac | "
          "switches/window |",
          "|---|---|---|---|---|---|---|"]
    for _, r in strat.iterrows():
        md.append(f"| {int(r.k)} | {int(r.n)} | {r.credit_carried:.3f} | "
                   f"{r.credit_background:.3f} | {r.margin:.3f} | {r.het_frac:.3f} | "
                   f"{r.mean_switches_per_window:.2f} |")
    md.append("")
    md.append("## Real simval 0.1x rows\n")
    md.append("| row | kind | k | credit (raw) carried/bg | credit (binary) carried/bg | "
               "ps4g carried/bg | cardinality | frac cells>1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in real_results:
        ps4g = (f"{r.get('ps4g_credit_carried', float('nan')):.3f} / "
                f"{r.get('ps4g_credit_background', float('nan')):.3f}"
                if "ps4g_credit_carried" in r else "n/a")
        md.append(f"| {r['row_id']} | {r['kind']} | {r['k']} | "
                   f"{r['credit_raw_carried']:.3f} / {r['credit_raw_background']:.3f} | "
                   f"{r['credit_bin_carried']:.3f} / {r['credit_bin_background']:.3f} | "
                   f"{ps4g} | {r['mean_cardinality']:.2f} | {r['frac_cells_gt1']:.4f} |")
    md.append("")
    (RESULTS_DIR / "support_profile.md").write_text("\n".join(md) + "\n")

    print(f"\nWrote {RESULTS_DIR/'support_profile.tsv'}")
    print(f"Wrote {RESULTS_DIR/'support_profile_hist.json'}")
    print(f"Wrote {RESULTS_DIR/'support_profile.md'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
