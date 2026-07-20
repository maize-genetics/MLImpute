"""
Diagnostic: does the diploid encoder force heterozygosity through the LOCAL
transition cost c, instead of via the homo_penalty crutch?

Over a het region with alternating single-gamete reads, the homozygous-FLIP decode
({A,A}<->{B,B}) maximizes emission and only loses to the stable het pair {A,B} when
c > (h-l)/2 sustained. So if the encoder is learning the right mechanism we expect,
on a no-penalty model:
  (1) mean c HIGHER at true-het loci (h1!=h2) than homozygous loci, and
  (2) the decode to actually PREDICT het (pred het-fraction ~ true het-fraction),
      not collapse to homozygous (pred het-fraction -> 0).

Reports both, stratified by per-individual inbreeding F. Compare pair_acc to the
penalty arms (eval_diploid_byF): hp0 outbred 0.171, hp3 0.445, adaptive-stable 0.445.

Usage:
  LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
    .pixi/envs/gpu/bin/python src/python/crf/diag_c_het.py \
      --ckpt <ckpt> --data /workdir/esb33/data/training/sim_e7_mixF.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_diploid import (GRITSCRFDiploid, make_diploid_splits,
                                       _dcrf_viterbi)

FBUCKETS = [(1.0, 1.01, "F=1 (inbred)"), (0.66, 1.0, "0.66-1"),
            (0.33, 0.66, "0.33-0.66"), (0.0, 0.33, "F<0.33 (outbred)"),
            (0.0, 1.01, "all")]


@torch.no_grad()
def collect(model, ds, device, bs):
    """Per-window arrays: c [.,T], pred-het 0/1 [.,T], true-het 0/1 [.,T],
    pair-correct 0/1 [.,T]."""
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    C, ph, th, pcorr = [], [], [], []
    for b in loader:
        X = b["input_embeds"].to(device)
        h1, h2 = b["h1"].to(device), b["h2"].to(device)
        emis_p, _, c = model(X, None)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        lo, hi = model.pi[pred], model.pj[pred]
        pair_true = model.pair_table[h1, h2]
        C.append(c.float().cpu().numpy())
        ph.append((lo != hi).cpu().numpy())                 # predicted het
        th.append((h1 != h2).cpu().numpy())                 # true het
        pcorr.append((pred == pair_true).cpu().numpy())
    return (np.concatenate(C), np.concatenate(ph),
            np.concatenate(th), np.concatenate(pcorr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--finb", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()

    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    _, _, test_ds = make_diploid_splits(args.data, args.num_parents,
                                        args.val_frac, args.test_frac)
    finb_path = args.finb or (Path(args.data).with_suffix("").as_posix() + ".finb.npy")
    finb = np.load(finb_path)
    N = finb.shape[0]
    n_test = int(N * args.test_frac)
    fte = finb[N - n_test:]                                  # per-window F (test)

    C, ph, th, pcorr = collect(model, test_ds, device, args.batch_size)
    assert len(C) == len(fte), f"{len(C)} != {len(fte)}"

    print(f"\n[diag c-by-het] {Path(args.ckpt).parent.name}  data={Path(args.data).name}")
    hdr = (f"{'F bucket':>16} {'win':>5} {'pair_acc':>8} "
           f"{'c@het':>7} {'c@homo':>7} {'c_ratio':>7} {'predHet':>7} {'trueHet':>7}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for lo, hi, lbl in FBUCKETS:
        m = (fte >= lo) & (fte < hi)
        if m.sum() == 0:
            continue
        c = C[m]; pe = ph[m]; tr = th[m]; pc = pcorr[m]
        het = tr.astype(bool); homo = ~het
        c_het = c[het].mean() if het.any() else float("nan")
        c_homo = c[homo].mean() if homo.any() else float("nan")
        ratio = c_het / c_homo if (homo.any() and c_homo != 0) else float("nan")
        print(f"{lbl:>16} {int(m.sum()):>5} {pc.mean():>8.4f} "
              f"{c_het:>7.3f} {c_homo:>7.3f} {ratio:>7.2f} "
              f"{pe.mean():>7.3f} {tr.mean():>7.3f}")
        rows.append(f"{lbl}: pair={pc.mean():.4f} c@het={c_het:.3f} c@homo={c_homo:.3f} "
                    f"predHet={pe.mean():.3f} trueHet={tr.mean():.3f}")

    print("\nread: if working -> c@het > c@homo (ratio>1) AND predHet ~ trueHet.")
    print("      if collapsed -> predHet << trueHet (homozygous-flip wins).")
    out = Path(args.workdir) / "results" / "e7_diag_c_het.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name}] {Path(args.data).name}\n  "
                + "\n  ".join(rows) + "\n")
    print(f"\nappended -> {out}")


if __name__ == "__main__":
    main()
