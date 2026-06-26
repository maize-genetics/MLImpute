"""
E7: diploid pair/hap accuracy stratified by per-individual inbreeding F.

A mixed-inbreeding panel (`simulate_alleles.py --mixed-inbreeding`, companion
`<data>.finb.npy`) contains fully inbred lines (F=1, identical gametes →
haploid-like, no phasing) and outbred individuals (F<1, interleaved single-gamete
reads that must be phased). This evaluator decodes a GRITSCRFDiploid checkpoint on
the test split and reports pair_acc / hap_acc bucketed by F, so we can see whether
ONE model handles the whole range (inbred near the haploid ceiling, outbred phased
above chance).

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_diploid_byF.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/sim_e7_mixF.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                       make_diploid_individual_splits, _dcrf_viterbi)

FBUCKETS = [(1.0, 1.01, "F=1 (inbred)"), (0.66, 1.0, "0.66-1"),
            (0.33, 0.66, "0.33-0.66"), (0.0, 0.33, "F<0.33")]


@torch.no_grad()
def per_window(model, ds, device, bs):
    """Per-window pair-correct (0/1) and hap-correct (0/1/2)."""
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    pc, hc = [], []
    for b in loader:
        X = b["input_embeds"].to(device)
        h1, h2 = b["h1"].to(device), b["h2"].to(device)
        scale = b.get("homo_scale")
        emis_p, _, c = model(X, scale.to(device) if scale is not None else None)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        pair_true = model.pair_table[h1, h2]
        pc.append((pred == pair_true).float().mean(1).cpu().numpy())       # [B] per-window
        lo, hi = model.pi[pred], model.pj[pred]
        tlo, thi = torch.minimum(h1, h2), torch.maximum(h1, h2)
        hc.append((((lo == tlo).float() + (hi == thi).float()) / 2).mean(1).cpu().numpy())
    return np.concatenate(pc), np.concatenate(hc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--finb", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adaptive", action="store_true",
                   help="model trained with --adaptive-homo (feeds per-individual scale)")
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    if args.adaptive:
        _, _, test_ds = make_diploid_individual_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual)
    else:
        _, _, test_ds = make_diploid_splits(args.data, args.num_parents,
                                            args.val_frac, args.test_frac)
    finb_path = args.finb or (Path(args.data).with_suffix("").as_posix() + ".finb.npy")
    finb = np.load(finb_path)
    N = finb.shape[0]
    n_test = int(N * args.test_frac)
    fte = finb[N - n_test:]                                  # test-split F per window

    pc, hc = per_window(model, test_ds, device, args.batch_size)
    if len(fte) != len(pc):
        raise ValueError(f"finb test {len(fte)} != windows {len(pc)}")

    print(f"\n[E7 diploid by inbreeding] {Path(args.ckpt).parent.name}  data={Path(args.data).name}")
    hdr = f"{'F bucket':>14} {'windows':>8} {'pair_acc':>9} {'hap_acc':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for lo, hi, lbl in FBUCKETS + [(0.0, 1.01, "all")]:
        m = (fte >= lo) & (fte < hi)
        if m.sum() == 0:
            continue
        print(f"{lbl:>14} {int(m.sum()):>8} {pc[m].mean():>9.4f} {hc[m].mean():>8.4f}")
        rows.append(f"{lbl}={pc[m].mean():.4f}/{hc[m].mean():.4f}")

    out = Path(args.workdir) / "results" / "e7_inbreeding.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name}] {Path(args.data).name} pair/hap by F: "
                + " | ".join(rows) + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
