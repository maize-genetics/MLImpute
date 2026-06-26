"""
E5 evaluator: does relatedness conditioning beat the per-window IBD ceiling?

The IBD ceiling (analyze_ibd_ceiling.py) is the best a *read-only*, per-window
decoder can do — within a window, IBD-identical founders are indistinguishable.
The E5 relatedness signal is cross-window (genome-wide founder affinity for the
individual), so it can legitimately exceed that ceiling. This script decodes a
trained checkpoint on the individual test split (feeding ext_emb when the model
was trained with --relatedness) and prints the same per-band table as the ceiling
analysis, so the relatedness arm's accuracy can be read directly against the
ceiling and against the no-relatedness baseline.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_e5.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/<sim>.npy \
        --windows-per-individual 100 --max-windows 20000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import GRITSCRFHaploid, make_individual_splits
from python.crf.analyze_ibd_ceiling import analyze, fmt


@torch.no_grad()
def predict(model, ds, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds = []
    for batch in loader:
        X = batch["input_embeds"].to(device)
        ext = batch.get("ext_emb")
        ext = ext.to(device) if ext is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X, ext)
        preds.append(model.decode(emis_f, c).cpu().numpy())
    return np.concatenate(preds, axis=0)


def ibd_test_slice(ibd_path, G, val_frac, test_frac, limit_n):
    """Mirror make_individual_splits' by-individual head slice for the IBD file."""
    ibd = np.load(ibd_path, mmap_mode="r")
    if limit_n:
        ibd = ibd[:(limit_n // G) * G]
    N = len(ibd)
    n_ind = N // G
    n_test = int(n_ind * test_frac) * G
    n_val = int(n_ind * val_frac) * G
    return np.asarray(ibd[N - n_test:])


def parse_args():
    p = argparse.ArgumentParser(description="E5 relatedness eval vs IBD ceiling")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--ibd", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    related = getattr(model, "ext_dim", 0) > 0

    _, _, test_ds = make_individual_splits(
        args.data, args.num_parents, args.val_frac, args.test_frac,
        args.windows_per_individual, limit_n=args.limit_n)

    ibd_path = args.ibd or (Path(args.data).with_suffix("").as_posix() + ".ibd.npy")
    ibd = ibd_test_slice(ibd_path, args.windows_per_individual,
                         args.val_frac, args.test_frac, args.limit_n)
    if len(ibd) != len(test_ds.data):
        raise ValueError(f"IBD rows {len(ibd)} != test rows {len(test_ds.data)}")

    K = args.num_parents
    arr = np.asarray(test_ds.data)
    feats = arr[:, :, :K].astype(np.int16)
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K)
    pred = predict(model, test_ds, device, args.batch_size, args.num_workers)

    stats = analyze(feats, h1, ibd.astype(np.int32), pred, K, args.max_windows)
    table = fmt(stats)
    tag = args.tag or Path(args.ckpt).parent.name
    arm = "relatedness" if related else "baseline(no-rel)"
    print(f"\n[{tag}] E5 {arm}  data={Path(args.data).name}  "
          f"(CRF acc vs read-only ceiling; relatedness may exceed ceiling)")
    print(table)

    out = Path(args.workdir) / "results" / "e5_eval.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{tag}] arm={arm} data={Path(args.data).name} ckpt={args.ckpt}\n")
        f.write(table + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
