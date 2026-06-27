"""
Export example diploid windows for the GRITS Tauri viewer (true vs predicted path).

For a curated set of test-split windows the viewer can open, this writes per window
the three files the heatmap + overlay loader expects (see crates/parser-core/src/
{ps4g,npy}.rs):

  window_<rank>_<class>_<het>/
    matrix.ps4g     # heatmap: each site t -> a position; gameteSet = founders
                    #          matching at t (reconstructs the founder-match matrix)
    observed.npy    # (T, K+2): rightmost 2 cols = true H1,H2  (label_cols = 2)
    pred.npy        # (T, 2): Viterbi-decoded predicted founders P1,P2

K gametes (F0..F{K-1}) are declared in the PS4G header so the viewer reports K
gametes and the observed-npy label-column math (label_cols = ncols - num_gametes = 2)
is exact. Predicted founders are clipped to [0,K-1] (the model's null/pad index K
does not occur in practice) so every path value maps to a real heatmap row.

Selection is a curated mix: for each of the six (founder-class x het-type) cells it
picks the highest- and lowest-accuracy windows, so wins and failures are both visible.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/export_windows.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/sim_breedpop.npy \
        --run-name e11-breedpop-learnedhet
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from python.crf.train_diploid import (GRITSCRFDiploid, _dcrf_viterbi,
                                       _founder_affinity)

CLASS_NAMES = {0: "k2-F2", 1: "k8-S1", 2: "outbred"}


@torch.no_grad()
def decode_test(model, feats_test, device, batch_size, ext_test=None):
    """Viterbi-decode every test window. Returns P1,P2 [n,T] founder indices.
    If ext_test [n,K,2] is given (model trained with --founder-affinity), it is
    fed as ext_emb so the decode reflects the founder-affinity prior."""
    n, T, _ = feats_test.shape
    P1 = np.empty((n, T), dtype=np.int64)
    P2 = np.empty((n, T), dtype=np.int64)
    for s in range(0, n, batch_size):
        X = torch.tensor(feats_test[s:s + batch_size], dtype=torch.float32, device=device)
        ext = None
        if ext_test is not None:
            ext = torch.tensor(ext_test[s:s + batch_size], dtype=torch.float32, device=device)
        emis_p, _, c = model(X, None, ext)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        P1[s:s + batch_size] = model.pi[pred].cpu().numpy()
        P2[s:s + batch_size] = model.pj[pred].cpu().numpy()
    return P1, P2


def write_ps4g(path, feats, contig):
    """feats [T,K] 0/1 -> PS4G text. Declares K gametes (F0..F{K-1}); one data
    row per site with gameteSet = founders matching at that site. Every site is
    emitted (empty gameteSet for zero-match sites) so num_positions == T."""
    T, K = feats.shape
    per_founder = feats.sum(0).astype(int)                      # [K] match totals
    lines = [f"#TotalUniqueCounts: {int(feats.sum())}",
             "#gamete\tgameteIndex\tcount"]
    for i in range(K):
        lines.append(f"#F{i}:0\t{i}\t{int(per_founder[i])}")
    lines.append("gameteSet\trefContig\trefPosBinned\tcount")
    for t in range(T):
        hits = np.nonzero(feats[t])[0]
        gset = ",".join(str(int(f)) for f in hits)
        lines.append(f"{gset}\t{contig}\t{t}\t1")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cls", default="")
    p.add_argument("--finb", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-per-cell", type=int, default=2,
                   help="windows per (class x het) cell, bracketing accuracy "
                        "(highest + lowest; >2 fills evenly between)")
    p.add_argument("--max-decode", type=int, default=3000,
                   help="random subset of test windows to Viterbi-decode for "
                        "selection (decoding all of them is needlessly slow)")
    p.add_argument("--out", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--founder-affinity", action="store_true",
                   help="ckpt trained with --founder-affinity; feed per-individual "
                        "ext_emb so the decode reflects the affinity prior")
    p.add_argument("--windows-per-individual", type=int, default=50,
                   help="individual block size for the affinity ext_emb (must match "
                        "training); the test slice must be a whole multiple of it")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRITSCRFDiploid.load_from_checkpoint(
        args.ckpt, map_location=device).eval().to(device)
    K = args.num_parents                                       # gametes = K founders

    data = np.load(args.data, allow_pickle=True, mmap_mode="r")
    N = len(data)
    n_test = int(N * args.test_frac)
    test = np.asarray(data[N - n_test:])                       # [n_test,T,C]
    feats = test[:, :, :K].astype(np.float32)                  # 0/1 match matrix
    h1 = np.clip(test[:, :, K].astype(np.int64), 0, K)
    h2 = np.clip(test[:, :, K + 1].astype(np.int64), 0, K)
    T = feats.shape[1]

    stem = Path(args.data).with_suffix("").as_posix()
    cls = np.load(args.cls or stem + ".cls.npy")[N - n_test:]
    finb = np.load(args.finb or stem + ".finb.npy")[N - n_test:]

    # Founder-affinity ext_emb is per-individual (mean match-rate over the
    # individual's G windows), so compute it on the FULL test slice — which is a
    # whole number of individuals — before any subsampling.
    ext = None
    if args.founder_affinity:
        G = args.windows_per_individual
        if n_test % G:
            raise ValueError(f"test windows {n_test} not a multiple of "
                             f"windows/ind {G}; affinity ext_emb would misalign")
        blk = feats.reshape(n_test // G, G, T, K)                  # [ind,G,T,K]
        aff = np.stack([_founder_affinity(blk[i]) for i in range(len(blk))])  # [ind,K,2]
        ext = np.repeat(aff, G, axis=0).astype(np.float32)         # [n_test,K,2]

    # Decode only a random subset for selection — the full test set is needlessly
    # slow and we only export a dozen windows.
    rng = np.random.default_rng(args.seed)
    if args.max_decode and args.max_decode < n_test:
        sub = np.sort(rng.choice(n_test, args.max_decode, replace=False))
        feats, h1, h2, cls, finb = feats[sub], h1[sub], h2[sub], cls[sub], finb[sub]
        if ext is not None:
            ext = ext[sub]

    P1, P2 = decode_test(model, feats, device, args.batch_size, ext)

    # per-window pair accuracy (order-insensitive) + het fractions
    pair_true = model.pair_table.cpu().numpy()[h1, h2]         # [n,T]
    pair_pred = model.pair_table.cpu().numpy()[
        np.minimum(P1, P2), np.maximum(P1, P2)]
    pair_acc = (pair_pred == pair_true).mean(1)                # [n_test]
    true_het = (h1 != h2).mean(1)
    pred_het = (P1 != P2).mean(1)

    run = args.run_name or Path(args.ckpt).parent.parent.name
    out = Path(args.out) if args.out else Path("/workdir/esb33/viewer_windows") / run
    out.mkdir(parents=True, exist_ok=True)

    rows, rank = [], 0
    for c in (0, 1, 2):
        for is_het, het_lbl in ((False, "inbred"), (True, "het")):
            cell = np.flatnonzero((cls == c) & ((finb > 0) if is_het else (finb == 0)))
            if cell.size == 0:
                continue
            order = cell[np.argsort(pair_acc[cell])]            # low -> high
            take = min(args.n_per_cell, order.size)
            # bracket: include the lowest and highest, fill evenly between
            picks = order[np.linspace(0, order.size - 1, take).round().astype(int)]
            for w in picks:
                tag = f"w{rank:02d}"
                folder = out / f"window_{rank:02d}_{CLASS_NAMES[c]}_{het_lbl}_acc{pair_acc[w]:.2f}"
                folder.mkdir(parents=True, exist_ok=True)
                write_ps4g(folder / "matrix.ps4g", feats[w], tag)
                # observed: K founder-match cols + true H1,H2 (label_cols = 2)
                obs = np.zeros((T, K + 2), dtype=np.int32)
                obs[:, :K] = feats[w].astype(np.int32)
                obs[:, K] = h1[w]
                obs[:, K + 1] = h2[w]
                np.save(folder / "observed.npy", obs)
                # predicted founders; the null/pad index K never occurs in practice
                # but clip defensively so every value maps to a real gamete row.
                pr = np.stack([P1[w], P2[w]], axis=1).astype(np.int32)
                np.save(folder / "pred.npy", np.clip(pr, 0, K - 1))
                rows.append((folder.name, CLASS_NAMES[c], het_lbl, pair_acc[w],
                             true_het[w], pred_het[w]))
                rank += 1

    man = ["# Exported windows — true vs predicted founder path",
           "",
           f"- ckpt: `{args.ckpt}`",
           f"- data: `{args.data}`  (test split, last {n_test} windows; T={T}, K={K})",
           f"- gametes declared: {K} (F0..F{K-1})",
           "",
           "Open per window in the viewer: `matrix.ps4g` (PS4G Explorer), then "
           "`observed.npy` + `pred.npy` via the Heatmap Observed/Predictions buttons. "
           "True paths draw solid, predicted dashed.",
           "",
           "| window | class | het | pair_acc | trueHet | predHet |",
           "|---|---|---|---:|---:|---:|"]
    for name, cn, hl, pa, th, ph in rows:
        man.append(f"| {name} | {cn} | {hl} | {pa:.3f} | {th:.3f} | {ph:.3f} |")
    (out / "MANIFEST.md").write_text("\n".join(man) + "\n")

    print(f"wrote {len(rows)} windows → {out}")
    print("\n".join(man[-len(rows) - 2:]))


if __name__ == "__main__":
    main()
