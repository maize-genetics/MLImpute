"""
E11: diploid pair/hap accuracy stratified by founder-count CLASS x het-type.

The breeding-population sim (`simulate_alleles.py --breeding-pop`) writes a
per-window class id (`<data>.cls.npy`: 0=k2 F2, 1=k8 S1, 2=outbred[12,24]) and a
per-window het target (`<data>.finb.npy`: 0 => inbred, >0 => het). This evaluator
decodes a GRITSCRFDiploid checkpoint on the test split and reports pair_acc /
hap_acc for each (class x inbred/het) cell, so we can see whether ONE model holds
across the whole breeding population -- especially the hard F2 (k=2 + 50% localized
het).

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_diploid_byclass.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/sim_breedpop.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                       make_diploid_affinity_splits)
from python.crf.eval_diploid_byF import per_window

CLASS_NAMES = {0: "k=2 F2", 1: "k=8 S1", 2: "outbred[12,24]"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cls", default="")
    p.add_argument("--finb", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--founder-affinity", action="store_true",
                   help="model trained with --founder-affinity (feed per-individual "
                        "ext_emb). Needs --windows-per-individual.")
    p.add_argument("--windows-per-individual", type=int, default=50)
    p.add_argument("--decode", choices=["viterbi", "marginal"], default="viterbi",
                   help="viterbi = MAP joint path; marginal = posterior per-site "
                        "argmax (forward-backward), optimal for per-site accuracy.")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(
        args.ckpt, map_location=device).eval().to(device)

    if args.founder_affinity:
        _, _, test_ds = make_diploid_affinity_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac,
            args.windows_per_individual)
    else:
        _, _, test_ds = make_diploid_splits(
            args.data, args.num_parents, args.val_frac, args.test_frac)

    stem = Path(args.data).with_suffix("").as_posix()
    cls = np.load(args.cls or stem + ".cls.npy")
    finb = np.load(args.finb or stem + ".finb.npy")
    N = cls.shape[0]
    n_test = int(N * args.test_frac)
    cte = cls[N - n_test:]
    fte = finb[N - n_test:]

    pc, hc = per_window(model, test_ds, device, args.batch_size, decode=args.decode)
    if len(cte) != len(pc):
        raise ValueError(f"cls test {len(cte)} != windows {len(pc)}")

    print(f"\n[E11 diploid by class x het] {Path(args.ckpt).parent.name}  "
          f"data={Path(args.data).name}  decode={args.decode}")
    hdr = f"{'class':>16} {'het-type':>9} {'windows':>8} {'pair_acc':>9} {'hap_acc':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for c in (0, 1, 2):
        for is_het, lbl in ((False, "inbred"), (True, "het")):
            m = (cte == c) & ((fte > 0) if is_het else (fte == 0))
            if m.sum() == 0:
                continue
            print(f"{CLASS_NAMES[c]:>16} {lbl:>9} {int(m.sum()):>8} "
                  f"{pc[m].mean():>9.4f} {hc[m].mean():>8.4f}")
            rows.append(f"{CLASS_NAMES[c]}/{lbl}={pc[m].mean():.4f}/{hc[m].mean():.4f}")
    print("-" * len(hdr))
    print(f"{'all':>16} {'':>9} {len(pc):>8} {pc.mean():>9.4f} {hc.mean():>8.4f}")
    rows.append(f"all={pc.mean():.4f}/{hc.mean():.4f}")

    out = Path(args.workdir) / "results" / "e11_breedpop.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name} decode={args.decode}] "
                f"{Path(args.data).name} pair/hap by class x het: "
                + " | ".join(rows) + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
