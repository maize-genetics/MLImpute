"""
E5 decisive table: does relatedness help, stratified by founders-per-individual?

The relatedness headroom (read-only ceiling -> relatedness ceiling) is largest for
individuals that descend from FEW founders: there, a carried founder is the active
source in a large fraction of windows, so its genome-wide affinity stands clearly
above background, and most confusable founders are out-of-set (rulable out). For
many-founder individuals the affinity contrast is weak and little is rulable out.
Aggregating over all k can therefore hide a real low-k benefit. This script decodes
the baseline and the relatedness checkpoints on the same test split and reports, per
founders-per-individual bucket: baseline acc, relatedness acc, read-only ceiling,
and relatedness ceiling.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_e5_byk.py \
        --base-ckpt <base.ckpt> --rel-ckpt <rel.ckpt> \
        --data /workdir/esb33/data/training/sim_e5_th6.npy --windows-per-individual 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import GRITSCRFHaploid, make_individual_splits

KBUCKETS = [(2, 4, "2-4"), (5, 8, "5-8"), (9, 16, "9-16"), (17, 24, "17-24")]


@torch.no_grad()
def predict(model, ds, device, bs=128):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    for b in loader:
        X = b["input_embeds"].to(device)
        ext = b.get("ext_emb")
        ext = ext.to(device) if ext is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X, ext)
        out.append(model.decode(emis_f, c).cpu().numpy())
    return np.concatenate(out, 0)


def ceilings(feats, h1, inset, w):
    """Per-window read-only and relatedness ceiling numerators (sites-weighted)."""
    h = h1[w]
    seg_id = np.concatenate([[0], np.cumsum(h[1:] != h[:-1])])
    fe, ins = feats[w], inset[w]
    rows = np.arange(len(h))
    read = rel = 0.0
    for seg in range(seg_id[-1] + 1):
        sl = rows[seg_id == seg]
        tf = h[sl[0]]
        eq = (fe[sl] == fe[sl, tf][:, None]).all(axis=0)
        read += sl.size / int(eq.sum())
        rel += sl.size / max(int((eq & ins).sum()), 1)
    return read, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--rel-ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--max-windows", type=int, default=10000)
    args = p.parse_args()
    K, G = args.num_parents, args.windows_per_individual
    device = torch.device("cuda")

    _, _, ds = make_individual_splits(args.data, K, 0.10, 0.10, G)
    arr = np.asarray(ds.data)
    feats = arr[:, :, :K].astype(np.int16)
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K - 1)
    n = min(args.max_windows, len(arr)) if args.max_windows else len(arr)

    # per-individual founder set (test windows are whole individuals, G contiguous)
    n_ind = len(arr) // G
    win_ind = np.repeat(np.arange(n_ind), G)
    sets = {i: np.unique(h1[win_ind == i]) for i in range(n_ind)}
    kof = np.array([len(sets[win_ind[w]]) for w in range(len(arr))])
    inset = np.zeros((len(arr), K), dtype=bool)
    for i, s in sets.items():
        inset[np.ix_(np.flatnonzero(win_ind == i), s)] = True

    base = GRITSCRFHaploid.load_from_checkpoint(args.base_ckpt, map_location=device).eval().to(device)
    rel = GRITSCRFHaploid.load_from_checkpoint(args.rel_ckpt, map_location=device).eval().to(device)
    pb = predict(base, ds, device)
    pr = predict(rel, ds, device)

    agg = {lbl: dict(sites=0, base=0, rel=0, rc=0.0, lc=0.0) for *_, lbl in KBUCKETS}
    agg["all"] = dict(sites=0, base=0, rel=0, rc=0.0, lc=0.0)
    for w in range(n):
        lbl = next(l for lo, hi, l in KBUCKETS if lo <= kof[w] <= hi)
        read, rl = ceilings(feats, h1, inset, w)
        for s in (agg[lbl], agg["all"]):
            s["sites"] += h1.shape[1]
            s["base"] += int((pb[w] == h1[w]).sum())
            s["rel"] += int((pr[w] == h1[w]).sum())
            s["rc"] += read
            s["lc"] += rl

    hdr = f"{'k(founders)':>11} {'sites':>9} {'base':>7} {'related':>8} {'read-ceil':>10} {'rel-ceil':>9} {'rel-base':>9}"
    print(f"\n[E5 by founders/individual] data={Path(args.data).name}")
    print(hdr); print("-" * len(hdr))
    for lbl in [l for *_, l in KBUCKETS] + ["all"]:
        s = agg[lbl]
        if not s["sites"]:
            continue
        b, r = s["base"] / s["sites"], s["rel"] / s["sites"]
        print(f"{lbl:>11} {s['sites']:>9,} {b:>7.4f} {r:>8.4f} "
              f"{s['rc']/s['sites']:>10.4f} {s['lc']/s['sites']:>9.4f} {r-b:>+9.4f}")


if __name__ == "__main__":
    main()
