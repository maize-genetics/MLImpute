"""
Tie-stratified diploid accuracy: does the model break within-window IBD ties?

At a site where several founders match the single read (an IBD tie), the local
emission is ambiguous — the model must use context (transitions) or a genome-wide
founder prior (--founder-affinity) to pick the right founder. This evaluator decodes
a checkpoint and reports per-site hap accuracy (partial credit, unordered) binned by
the number of founders matching at the site. Run it on the learned-het baseline and
the +affinity model: if affinity is working, the gain concentrates in the high-tie
bins.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_diploid_ties.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/sim_breedpop.npy \
        [--founder-affinity]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                       make_diploid_affinity_splits, _dcrf_viterbi)

BINS = [(1, 1, "1 (no tie)"), (2, 3, "2-3"), (4, 6, "4-6"),
        (7, 12, "7-12"), (13, 999, "13+")]


@torch.no_grad()
def tie_accuracy(model, ds, device, bs, K):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    corr = {b[2]: 0.0 for b in BINS}
    tot = {b[2]: 0 for b in BINS}
    for b in loader:
        X = b["input_embeds"].to(device)
        h1, h2 = b["h1"].to(device), b["h2"].to(device)
        ext = b.get("ext_emb")
        emis_p, _, c = model(X, b.get("homo_scale"),
                             ext.to(device) if ext is not None else None)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        lo, hi = model.pi[pred], model.pj[pred]
        tlo, thi = torch.minimum(h1, h2), torch.maximum(h1, h2)
        site_acc = ((lo == tlo).float() + (hi == thi).float()) / 2   # [B,T] in {0,.5,1}
        nmatch = X[:, :, :K].sum(-1)                                 # founders matching
        for lo_b, hi_b, name in BINS:
            m = (nmatch >= lo_b) & (nmatch <= hi_b)
            corr[name] += site_acc[m].sum().item()
            tot[name] += int(m.sum().item())
    return corr, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--founder-affinity", action="store_true")
    p.add_argument("--windows-per-individual", type=int, default=50)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(
        args.ckpt, map_location=device).eval().to(device)
    K = args.num_parents

    if args.founder_affinity:
        _, _, test_ds = make_diploid_affinity_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual)
    else:
        _, _, test_ds = make_diploid_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac)

    corr, tot = tie_accuracy(model, test_ds, device, args.batch_size, K)

    print(f"\n[tie-stratified hap acc] {Path(args.ckpt).parent.name}  "
          f"data={Path(args.data).name}  affinity={args.founder_affinity}")
    hdr = f"{'founders matching':>18} {'sites':>12} {'%sites':>7} {'hap_acc':>8}"
    print(hdr); print("-" * len(hdr))
    grand = sum(tot.values())
    rows = []
    for _, _, name in BINS:
        if tot[name] == 0:
            continue
        acc = corr[name] / tot[name]
        print(f"{name:>18} {tot[name]:>12,} {100*tot[name]/grand:>6.1f}% {acc:>8.4f}")
        rows.append(f"{name}={acc:.4f}")
    allacc = sum(corr.values()) / grand
    print(f"{'all':>18} {grand:>12,} {'100.0%':>7} {allacc:>8.4f}")

    out = Path(args.workdir) / "results" / "e11_ties.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name} affinity={args.founder_affinity}] "
                f"hap by tie: " + " | ".join(rows) + f" | all={allacc:.4f}\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
