"""
Pretty-print one window of a simulated/real input matrix with its label columns,
so the data layout can be eyeballed.

Layout (simulate_alleles.py):
  data .npy  [N, T, C]   cols 0:K = founder allele-match (0/1)
                         col K     = H1 true founder path
                         col K+1   = H2 true founder path (diploid only)
                         col K+2   = per-site true recomb rate (eval-only; if present)
  <data>.ind.npy   [N]        per-window individual id
  <data>.finb.npy  [N]        per-window inbreeding F (mixed-inbreeding sims)
  <data>.ibd.npy   [N,T,K]    per-site IBD lineage label per founder

A read at a site is ONE gamete's mini-haplotype, so the match row shows which
founders that single read is consistent with — not which gamete it came from.
The H1#/H2# markers show whether the true H1 / H2 founder matched that read.

Usage:
  PYTHONPATH=src python src/python/crf/show_window.py \
      --data /workdir/esb33/data/training/sim_outbred_k23.npy --window 0 --sites 32
"""

import argparse
from pathlib import Path

import numpy as np

HIT, MISS = "#", "."


def companion(data_path, suffix):
    p = Path(data_path).with_suffix("").as_posix() + suffix
    return np.load(p, mmap_mode="r") if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--sites", type=int, default=32, help="rows (sites) to show")
    ap.add_argument("--start", type=int, default=0, help="first site to show")
    ap.add_argument("--num-parents", type=int, default=24)
    args = ap.parse_args()

    K = args.num_parents
    data = np.load(args.data, mmap_mode="r")
    N, T, C = data.shape
    diploid = C >= K + 2
    has_rate = C >= K + 3
    w = args.window
    win = np.asarray(data[w]).astype(int)             # [T, C]
    match = win[:, :K]
    h1 = win[:, K]
    h2 = win[:, K + 1] if diploid else None
    rate = win[:, K + 2] if has_rate else None

    ind = companion(args.data, ".ind.npy")
    finb = companion(args.data, ".finb.npy")
    ibd = companion(args.data, ".ibd.npy")

    # ---- header ----
    founders = np.unique(np.concatenate([h1, h2]) if diploid else h1)
    founders = founders[(founders >= 0) & (founders < K)]
    print(f"\nfile   : {Path(args.data).name}   shape={data.shape}  dtype={data.dtype}")
    print(f"window : {w} / {N}   T={T} sites   {'DIPLOID' if diploid else 'HAPLOID'}  K={K}")
    line = f"founders in window: {set(founders.tolist())}  (k={len(founders)})"
    if ind is not None:
        line += f"   | individual={int(ind[w])}"
    if finb is not None:
        line += f"   | F={float(finb[w]):.2f}"
    print(line)
    print(f"glyph: '{HIT}'=read matches founder, '{MISS}'=no match"
          + ("   | H1#/H2# = true founder matched this read" if diploid else ""))

    # ---- founder index ruler ----
    tens = "".join(str((k // 10) % 10) if k % 10 == 0 else " " for k in range(K))
    units = "".join(str(k % 10) for k in range(K))
    pad = " " * 8
    extra = "  H1 H2  flags" if diploid else "  H1"
    print(f"\n{pad}{tens}")
    print(f"{'site':>6}  {units}{extra}")
    print(f"{pad}{'-'*K}")

    s0, s1 = args.start, min(args.start + args.sites, T)
    n_h1 = n_h2 = n_both = n_neither = 0
    for t in range(s0, s1):
        row = "".join(HIT if match[t, k] else MISS for k in range(K))
        if diploid:
            m1, m2 = match[t, h1[t]] == 1, match[t, h2[t]] == 1
            n_h1 += m1; n_h2 += m2; n_both += (m1 and m2); n_neither += (not m1 and not m2)
            flag = ("H1" + (HIT if m1 else MISS)) + " " + ("H2" + (HIT if m2 else MISS))
            ibd_note = ""
            if ibd is not None and ibd[w, t, h1[t]] == ibd[w, t, h2[t]]:
                ibd_note = "  <IBD>"            # H1 & H2 founders share a lineage here
            print(f"{t:>6}  {row}  {h1[t]:>2} {h2[t]:>2}  {flag}{ibd_note}")
        else:
            print(f"{t:>6}  {row}  {h1[t]:>2}")

    # ---- per-window coverage summary (diploid) ----
    if diploid:
        m1 = (match[np.arange(T), h1] == 1)
        m2 = (match[np.arange(T), h2] == 1)
        tot = T
        print(f"\nwindow coverage over all {T} sites:")
        print(f"  read matches H1 founder : {m1.mean():.3f}")
        print(f"  read matches H2 founder : {m2.mean():.3f}")
        print(f"  matches BOTH            : {(m1 & m2).mean():.3f}   (het site, both founders seen)")
        print(f"  matches NEITHER         : {(~m1 & ~m2).mean():.3f}   (dropout/error/IBD-elsewhere)")
        het = (h1 != h2).mean()
        print(f"  true het fraction (H1!=H2): {het:.3f}")


if __name__ == "__main__":
    main()
