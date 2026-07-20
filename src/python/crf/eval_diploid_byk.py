"""
E7 follow-up: diploid pair/hap accuracy stratified by an individual's FOUNDER
COUNT (k = number of distinct founders contributing to it), for outbred panels.

Motivation: in `sim_e7_mixF` founder-count and inbreeding are confounded (the
few-founder individuals are all inbred lines), so the few-founder *outbred* case
— an F1/F2-style cross of 2-3 founders, heterozygous — cannot be isolated. This
evaluator runs on a dedicated outbred sim (`--inbreeding 0` `--min/max-founders`)
where every individual is outbred, and reports accuracy by k. The model never saw
this sim, so the whole thing is held out.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_diploid_byk.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/sim_outbred_k23.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from python.crf.train_diploid import GRITSCRFDiploid, DiploidIndividualDataset
from python.crf.eval_diploid_byF import per_window


def founder_count_per_window(data, K, G):
    """k for each window = #distinct founders (H1 ∪ H2, excl. unknown=K) in its
    individual's G-window block."""
    N = len(data)
    kw = np.empty(N, dtype=np.int64)
    for i in range(N // G):
        sl = slice(i * G, (i + 1) * G)
        h1 = np.asarray(data[sl, :, K]).astype(np.int64).ravel()
        h2 = np.asarray(data[sl, :, K + 1]).astype(np.int64).ravel()
        f = np.unique(np.concatenate([h1, h2]))
        f = f[(f >= 0) & (f < K)]
        kw[sl] = len(f)
    return kw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adaptive", action="store_true", default=True)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()

    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    K, G = args.num_parents, args.windows_per_individual
    data = np.load(args.data, allow_pickle=True, mmap_mode="r")
    ds = DiploidIndividualDataset(data, K, G)
    kw = founder_count_per_window(data, K, G)

    pc, hc = per_window(model, ds, device, args.batch_size)
    assert len(pc) == len(kw), f"{len(pc)} != {len(kw)}"

    print(f"\n[E7 outbred by founder-count] {Path(args.ckpt).parent.name}  data={Path(args.data).name}")
    hdr = f"{'k founders':>12} {'inds':>6} {'windows':>8} {'pair_acc':>9} {'hap_acc':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    buckets = [(2, 2, "k=2"), (3, 3, "k=3"), (2, 3, "k=2-3"), (0, 999, "all")]
    for lo, hi, lbl in buckets:
        m = (kw >= lo) & (kw <= hi)
        if m.sum() == 0:
            continue
        ninds = int(m.sum() // G)
        print(f"{lbl:>12} {ninds:>6} {int(m.sum()):>8} {pc[m].mean():>9.4f} {hc[m].mean():>8.4f}")
        rows.append(f"{lbl}={pc[m].mean():.4f}/{hc[m].mean():.4f}")

    out = Path(args.workdir) / "results" / "e7_byfounders.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name}] {Path(args.data).name} pair/hap by k: "
                + " | ".join(rows) + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
