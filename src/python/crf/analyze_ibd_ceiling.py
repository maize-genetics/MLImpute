"""
IBD-ceiling analysis for low-recombination ([0,2]-bp / 512) windows.

Question: in low-recombination windows the haploid CRF tops out around 60-80%
founder accuracy.  Are the residual errors *IBD confusion* — the true founder is
identical-by-descent with one or more other founders across the whole block, so
their reads are bit-identical and no read-only decoder can separate them?  If so,
that accuracy is at/near the ceiling reachable without outside information
(relatedness / genome-wide founder presence).

This needs a coalescent sim built with `simulate_alleles.py --sharing-theta ...`,
which writes the per-site IBD lineage labels to `<data>.ibd.npy`.  For each
constant-founder segment we form the *indistinguishable set* S:

  * feature-based  S_feat = founders whose K-wide feature column is bit-identical
    to the true founder's over every site of the segment (exactly what a
    read-only decoder sees → the information-theoretic ceiling).
  * IBD-based      S_ibd  = founders on the true founder's IBD lineage at every
    site of the segment (the *mechanism*: why those columns are identical).

Ceiling = sites-weighted mean of 1/|S_feat| (best decoder picks the truth
uniformly from its indistinguishable set).  Each CRF error is then labelled
IBD-confusable (pred in S_feat → unavoidable) or genuine (pred not in S_feat →
recoverable signal the model missed).  Reported per true-breakpoint band.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src \
      .pixi/envs/gpu/bin/python src/python/crf/analyze_ibd_ceiling.py \
        --ckpt <path.ckpt> --data /workdir/esb33/data/training/<sim>.npy \
        --limit-n 0 --split test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import GRITSCRFHaploid, make_splits

# True-breakpoint bands (by switch count of the true founder path per window).
BANDS = [(0, 0, "0 bp"), (1, 2, "1-2 bp"), (3, 5, "3-5 bp"), (6, 10**9, "6+ bp")]


@torch.no_grad()
def predict(model, ds, device, batch_size, num_workers):
    """Viterbi founder path per window, in dataset order (no shuffle)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds = []
    for batch in loader:
        X = batch["input_embeds"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X)
        preds.append(model.decode(emis_f, c).cpu().numpy())
    return np.concatenate(preds, axis=0)


def ibd_test_slice(ibd_path, limit_n, val_frac, test_frac):
    """Reproduce make_splits' head-slice + contiguous test slice for the IBD file."""
    ibd = np.load(ibd_path, mmap_mode="r")
    if limit_n and limit_n < len(ibd):
        ibd = ibd[:limit_n]
    N = len(ibd)
    n_test = int(N * test_frac)
    n_val = int(N * val_frac)
    n_train = N - n_val - n_test
    return np.asarray(ibd[n_train + n_val:])


def analyze(feats, h1, lineage, pred, K, max_windows):
    """Per-segment indistinguishable sets, ceiling, and error decomposition.

    feats   [N,T,K] int   per-founder match indicator (0/1)
    h1      [N,T]   int    true founder path
    lineage [N,T,K] int    per-site IBD lineage label per founder
    pred    [N,T]   int    CRF Viterbi founder path
    Returns a dict keyed by band label, plus 'all'.
    """
    N, T = h1.shape
    if max_windows and max_windows < N:
        N = max_windows
    rows = np.arange(T)

    def fresh():
        return dict(sites=0, correct=0, ceil_num=0.0, err=0, err_ibd=0,
                    S_sum=0.0, S_seg=0, multi_sites=0, windows=0)
    stats = {lbl: fresh() for *_, lbl in BANDS}
    stats["all"] = fresh()

    for w in range(N):
        h = h1[w].astype(np.int64)
        switches = int((h[1:] != h[:-1]).sum())
        seg_id = np.concatenate([[0], np.cumsum(h[1:] != h[:-1])])
        lab = next(lbl for lo, hi, lbl in BANDS if lo <= switches <= hi)
        targets = (stats[lab], stats["all"])
        for s in targets:
            s["windows"] += 1

        fe, lin, pr = feats[w], lineage[w], pred[w]
        for seg in range(seg_id[-1] + 1):
            mask = seg_id == seg
            sl = rows[mask]
            tf = h[sl[0]]                                   # true founder this segment
            L = sl.size
            # feature-indistinguishable: identical column over the whole segment
            eq_feat = (fe[sl] == fe[sl, tf][:, None]).all(axis=0)   # [K]
            eq_ibd = (lin[sl] == lin[sl, tf][:, None]).all(axis=0)  # [K]
            Sf = int(eq_feat.sum())                          # >= 1 (truth itself)
            inv = 1.0 / Sf
            err_mask = pr[sl] != tf
            err = int(err_mask.sum())
            # an error is IBD-confusable if the predicted founder is itself in the
            # true founder's indistinguishable set (per-site predicted founder).
            err_ibd = int(eq_feat[pr[sl][err_mask]].sum()) if err else 0
            for s in targets:
                s["sites"] += L
                s["correct"] += L - err
                s["ceil_num"] += L * inv
                s["err"] += err
                s["err_ibd"] += err_ibd
                s["S_sum"] += Sf
                s["S_seg"] += 1
                if Sf > 1:
                    s["multi_sites"] += L
    return stats


def fmt(stats):
    lines = []
    hdr = (f"{'band':>8} {'win':>6} {'sites':>9} {'CRF acc':>8} {'ceiling':>8} "
           f"{'gap':>7} {'meanS':>6} {'multiS%':>8} {'errIBD%':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    order = [lbl for *_, lbl in BANDS] + ["all"]
    for lbl in order:
        s = stats[lbl]
        if s["sites"] == 0:
            continue
        acc = s["correct"] / s["sites"]
        ceil = s["ceil_num"] / s["sites"]
        meanS = s["S_sum"] / s["S_seg"]
        multi = 100 * s["multi_sites"] / s["sites"]
        errp = 100 * s["err_ibd"] / s["err"] if s["err"] else float("nan")
        lines.append(f"{lbl:>8} {s['windows']:>6} {s['sites']:>9,} {acc:>8.4f} "
                     f"{ceil:>8.4f} {acc - ceil:>+7.4f} {meanS:>6.2f} "
                     f"{multi:>7.1f}% {errp:>7.1f}%")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="IBD-ceiling analysis")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--ibd", default="", help="IBD-truth .npy (default <data>.ibd.npy)")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=0,
                   help="Cap analyzed windows for speed (0 = all of the split).")
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")

    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    train_ds, val_ds, test_ds = make_splits(
        args.data, args.num_parents, args.val_frac, args.test_frac,
        limit_n=args.limit_n)
    ds = test_ds if args.split == "test" else val_ds

    ibd_path = args.ibd or (Path(args.data).with_suffix("").as_posix() + ".ibd.npy")
    ibd = ibd_test_slice(ibd_path, args.limit_n, args.val_frac, args.test_frac)
    if args.split == "val":
        raise SystemExit("ibd slice currently wired for --split test only")
    if len(ibd) != len(ds.data):
        raise ValueError(f"IBD rows {len(ibd)} != test rows {len(ds.data)}; "
                         "data and .ibd.npy out of sync")

    K = args.num_parents
    arr = np.asarray(ds.data)
    feats = arr[:, :, :K].astype(np.int16)
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K)
    pred = predict(model, ds, device, args.batch_size, args.num_workers)

    stats = analyze(feats, h1, ibd.astype(np.int32), pred, K, args.max_windows)
    table = fmt(stats)

    tag = args.tag or Path(args.ckpt).parent.name
    print(f"\n[{tag}] IBD-ceiling  data={Path(args.data).name}  split={args.split}")
    print(table)

    res_dir = Path(args.workdir) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / "ibd_ceiling.txt"
    with open(out, "a") as f:
        f.write(f"\n[{tag}] data={Path(args.data).name} ckpt={args.ckpt}\n")
        f.write(table + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
