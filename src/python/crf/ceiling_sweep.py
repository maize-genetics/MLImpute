"""
Ceiling-vs-relatedness sweep (no model needed).

The IBD ceiling — the best founder accuracy any read-only decoder can reach — is
a property of the *data*, not the model: for each constant-founder segment it is
1/|S_feat|, where S_feat is the set of founders whose read column is bit-identical
to the truth across the whole segment (`analyze_ibd_ceiling.analyze`).  So we can
map how the ceiling moves with the coalescent relatedness parameter theta WITHOUT
training anything: generate a small sim at each theta and compute the ceiling
directly (here we pass the true path as the "prediction", so CRF-acc=1 / errIBD is
meaningless — only the ceiling / meanS / multiS% columns are read).

Larger theta = less relatedness (more singleton founders, smaller indistinguishable
sets) = higher ceiling.  Smaller theta = more IBD sharing = lower ceiling.

Usage:
    PYTHONPATH=src .pixi/envs/gpu/bin/python src/python/crf/ceiling_sweep.py \
        --data /workdir/esb33/data/training/sim_ewens_th6.npy --max-windows 8000
"""

import argparse
from pathlib import Path

import numpy as np

from python.crf.analyze_ibd_ceiling import analyze, fmt


def parse_args():
    p = argparse.ArgumentParser(description="IBD-ceiling vs theta (data-only)")
    p.add_argument("--data", required=True)
    p.add_argument("--ibd", default="", help="default <data>.ibd.npy")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--max-windows", type=int, default=8000)
    p.add_argument("--tag", default="")
    p.add_argument("--workdir", default="/workdir/esb33")
    return p.parse_args()


def main():
    args = parse_args()
    K = args.num_parents
    data = np.load(args.data, mmap_mode="r")
    n = min(args.max_windows, len(data)) if args.max_windows else len(data)
    arr = np.asarray(data[:n])
    ibd_path = args.ibd or (Path(args.data).with_suffix("").as_posix() + ".ibd.npy")
    ibd = np.asarray(np.load(ibd_path, mmap_mode="r")[:n]).astype(np.int32)

    feats = arr[:, :, :K].astype(np.int16)
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K)
    stats = analyze(feats, h1, ibd, h1, K, 0)   # pred=truth → ceiling-only read

    tag = args.tag or Path(args.data).stem
    table = fmt(stats)
    print(f"\n[{tag}] ceiling  data={Path(args.data).name}  windows={n}")
    print(table)

    res = Path(args.workdir) / "results" / "ceiling_sweep.txt"
    res.parent.mkdir(parents=True, exist_ok=True)
    with open(res, "a") as f:
        f.write(f"\n[{tag}] data={Path(args.data).name} windows={n}\n")
        f.write(table + "\n")
    print(f"\nappended → {res}")


if __name__ == "__main__":
    main()
