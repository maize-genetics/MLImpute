"""
E6: SNP-level imputation accuracy from the decoded founder path.

GRITS decodes a founder PATH; standard imputers report per-SNP genotype
concordance and dosage r². We bridge the two by composing the decoded path with
the founder × SNP allele panel (`<data>.panel.npy`, written by
`simulate_alleles.py --emit-snp-panel`):

    predicted_allele(t, l) = panel[t, decoded_founder(t), l]
    true_allele(t, l)      = panel[t, true_founder(t),   l]

Haploid (inbred, F=1): per-SNP genotype concordance. We also report founder-PATH
accuracy on the same sites so the path-vs-SNP gap is explicit — that gap is the
IBD effect: where the path errs it errs onto an IBD-confusable founder that shares
the allele, so the SNP call is usually still right. Concordance and (haploid)
allele r² are stratified by minor-allele frequency.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_snp.py \
        --ckpt <ckpt> --data /workdir/esb33/data/training/<sim>.npy \
        --windows-per-individual 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import (GRITSCRFHaploid, make_splits,
                                       make_individual_splits)

MAF_BINS = [(0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 0.20, "5-20%"),
            (0.20, 0.50, "20-50%")]


@torch.no_grad()
def predict(model, ds, device, bs):
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


def panel_test_slice(panel_path, related, G, val_frac, test_frac, limit_n):
    panel = np.load(panel_path, mmap_mode="r")               # [N,T,K,L]
    if related:
        if limit_n:
            panel = panel[:(limit_n // G) * G]
        N = len(panel)
        n_ind = N // G
        n_test = int(n_ind * test_frac) * G
    else:
        if limit_n and limit_n < len(panel):
            panel = panel[:limit_n]
        N = len(panel)
        n_test = int(N * test_frac)
    return np.asarray(panel[N - n_test:])


def main():
    p = argparse.ArgumentParser(description="E6 SNP-level imputation accuracy")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--panel", default="")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=0,
                   help=">0 if the model/sim use E5 individual grouping")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    K = args.num_parents
    device = torch.device("cuda")
    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    related = args.windows_per_individual > 0
    if related:
        _, _, ds = make_individual_splits(args.data, K, args.val_frac, args.test_frac,
                                          args.windows_per_individual, limit_n=args.limit_n)
    else:
        _, _, ds = make_splits(args.data, K, args.val_frac, args.test_frac,
                               limit_n=args.limit_n)

    panel_path = args.panel or (Path(args.data).with_suffix("").as_posix() + ".panel.npy")
    panel = panel_test_slice(panel_path, related, args.windows_per_individual,
                             args.val_frac, args.test_frac, args.limit_n)
    arr = np.asarray(ds.data)
    if len(panel) != len(arr):
        raise ValueError(f"panel rows {len(panel)} != test rows {len(arr)}")

    n = min(args.max_windows, len(arr)) if args.max_windows else len(arr)
    panel, arr = panel[:n], arr[:n]
    h1 = np.clip(arr[:, :, K].astype(np.int64), 0, K - 1)
    pred = predict(model, ds, device, args.batch_size)[:n]

    N, T, _, L = panel.shape
    wi = np.arange(N)[:, None]
    ti = np.arange(T)[None, :]
    true_allele = panel[wi, ti, h1]                          # [N,T,L]
    pred_allele = panel[wi, ti, pred]                        # [N,T,L]
    # derived-allele freq per SNP slot across all sites (proxy MAF for stratification)
    af = true_allele.reshape(-1, L).mean(0)                  # [L] mean over all sites
    # per-site, per-SNP correctness
    snp_ok = (true_allele == pred_allele)                    # [N,T,L]
    path_ok = (pred == h1)                                   # [N,T]

    # Stratify by per-SNP-column MAF (each of the L read slots has its own freq).
    print(f"\n[E6 SNP accuracy] data={Path(args.data).name}  ckpt={Path(args.ckpt).name}")
    print(f"  windows={N:,}  sites/window={T}  SNPs/read(L)={L}")
    path_acc = path_ok.mean()
    snp_acc = snp_ok.mean()
    print(f"  founder-PATH accuracy : {path_acc:.4f}")
    print(f"  SNP genotype concord. : {snp_acc:.4f}   (gap +{snp_acc - path_acc:.4f} = IBD-invisible-at-SNP)")

    # Per-site overall MAF stratification: bin SNP observations by their column freq.
    colmaf = np.minimum(af, 1 - af)                          # [L]
    hdr = f"{'MAF bin':>8} {'#SNP-cols':>9} {'concord':>8} {'allele-r2':>9}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))
    for lo, hi, lbl in MAF_BINS:
        cols = np.flatnonzero((colmaf >= lo) & (colmaf < hi))
        if cols.size == 0:
            continue
        t = true_allele[:, :, cols].reshape(-1).astype(np.float64)
        pr = pred_allele[:, :, cols].reshape(-1).astype(np.float64)
        conc = (t == pr).mean()
        if t.std() > 0 and pr.std() > 0:
            r2 = np.corrcoef(t, pr)[0, 1] ** 2
        else:
            r2 = float("nan")
        print(f"  {lbl:>8} {cols.size:>9} {conc:>8.4f} {r2:>9.4f}")

    out = Path(args.workdir) / "results" / "e6_snp.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name}] data={Path(args.data).name} "
                f"path_acc={path_acc:.4f} snp_concord={snp_acc:.4f}\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
