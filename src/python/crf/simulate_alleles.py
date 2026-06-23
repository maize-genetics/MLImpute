"""
Simulate shared-allele patterns for GRITS founder-path imputation.

Generates a well-posed synthetic imputation problem: each window is a piecewise
constant founder path (the recombination mosaic) observed only through a noisy
allele-sharing pattern.  The model must recover the path from temporal
continuity plus the fact that the true founder almost always shares the
sample's allele.

Per site the K-wide feature vector is a binary allele-match pattern:
  feature[f] = 1  if founder f carries the same allele as the sample, else 0.
The true founder is forced to 1 on "good" sites; on "bad" sites the pattern is
left fully random so the true founder need not match (genotyping error).
Average allele sharing p sets the rate at which *other* founders also match,
q = (p*K - 1) / (K - 1), so the expected fraction of matching founders is p.

Inbreeding coefficient F controls haplotype identity: with probability F the two
haplotypes (H1, H2) share one path (fully inbred when F=1); otherwise H2 is an
independent path and a founder matches if it equals either H1 or H2.

Output  (matches LabeledDatasetDiploid):
    NPY int8 tensor [windows, sites, founders + 2]
      cols 0:K   allele-match features
      col  K     H1 founder label  (0..K-1)
      col  K+1   H2 founder label  (0..K-1)

Usage:
    pixi run -- python src/python/crf/simulate_alleles.py \
        --workdir /workdir/esb33 --windows 100000
"""

import argparse
from pathlib import Path

import numpy as np


def _build_paths(rng, n, T, K, n_cross):
    """Piecewise-constant founder paths [n, T] with exactly n_cross switches each.

    n_cross is a [n] array; windows are grouped by switch count so the whole
    thing stays vectorized.  Adjacent segments are guaranteed to differ.
    """
    path = np.empty((n, T), dtype=np.int64)
    for v in np.unique(n_cross):
        rows = np.flatnonzero(n_cross == v)
        nv = rows.size

        # v distinct breakpoints in [1, T-1] per window
        bp = np.argsort(rng.random((nv, T - 1)), axis=1)[:, :v] + 1
        bp.sort(axis=1)

        # v+1 segment founders, adjacent guaranteed distinct
        seg = np.empty((nv, v + 1), dtype=np.int64)
        seg[:, 0] = rng.integers(0, K, nv)
        for s in range(1, v + 1):
            step = rng.integers(1, K, nv)          # 1..K-1 -> never repeats prev
            seg[:, s] = (seg[:, s - 1] + step) % K

        # segment index at each site = #breakpoints <= t
        sw = np.zeros((nv, T), dtype=np.int32)
        np.add.at(sw, (np.repeat(np.arange(nv), v), bp.ravel()), 1)
        seg_idx = np.cumsum(sw, axis=1)            # 0..v
        path[rows] = np.take_along_axis(seg, seg_idx, axis=1)
    return path


def simulate(rng, windows, sites, founders, min_cross, max_cross,
             inbreeding, allele_sharing, bad_frac, chunk=2000):
    K = founders
    T = sites
    q = (allele_sharing * K - 1.0) / (K - 1)       # match rate for non-true founders
    if q < 0:
        raise ValueError(
            f"allele_sharing={allele_sharing} too low for K={K}; "
            f"minimum is {1.0 / K:.4f}")

    out = np.empty((windows, T, K + 2), dtype=np.int8)

    for start in range(0, windows, chunk):
        n = min(chunk, windows - start)
        n_cross = rng.integers(min_cross, max_cross + 1, n)
        h1 = _build_paths(rng, n, T, K, n_cross)

        # Second haplotype: identical with prob F (inbred), else independent
        inbred = rng.random(n) < inbreeding
        h2 = h1.copy()
        outbred = np.flatnonzero(~inbred)
        if outbred.size:
            h2[outbred] = _build_paths(
                rng, outbred.size, T, K,
                rng.integers(min_cross, max_cross + 1, outbred.size))

        # Random allele-match background
        feats = (rng.random((n, T, K)) < q).astype(np.int8)

        # On good sites force the true founder(s) to match
        good = rng.random((n, T)) >= bad_frac
        ii = np.arange(n)[:, None]
        tt = np.arange(T)[None, :]
        cur = feats[ii, tt, h1]
        feats[ii, tt, h1] = np.where(good, 1, cur)
        cur = feats[ii, tt, h2]
        feats[ii, tt, h2] = np.where(good, 1, cur)

        out[start:start + n, :, :K] = feats
        out[start:start + n, :, K] = h1.astype(np.int8)
        out[start:start + n, :, K + 1] = h2.astype(np.int8)

    return out


def parse_args():
    p = argparse.ArgumentParser(description="Simulate shared-allele patterns")
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--out", default="sim_alleles.npy",
                   help="Filename written under <workdir>/data/training/")
    p.add_argument("--founders", type=int, default=24)
    p.add_argument("--sites", type=int, default=512, help="Site window length")
    p.add_argument("--windows", type=int, default=100000, help="Number of windows")
    p.add_argument("--min-crossovers", type=int, default=2)
    p.add_argument("--max-crossovers", type=int, default=10)
    p.add_argument("--inbreeding", type=float, default=1.0,
                   help="Inbreeding coefficient F in [0,1]; P(H1==H2 path)")
    p.add_argument("--allele-sharing", type=float, default=0.2,
                   help="Average fraction of founders sharing the sample allele")
    p.add_argument("--bad-frac", type=float, default=0.05,
                   help="Proportion of sites with corrupted (random) patterns")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data = simulate(
        rng, args.windows, args.sites, args.founders,
        args.min_crossovers, args.max_crossovers,
        args.inbreeding, args.allele_sharing, args.bad_frac)

    out_dir = Path(args.workdir) / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    np.save(out_path, data)

    # Verification summary
    K = args.founders
    feats = data[:, :, :K]
    h1 = data[:, :, K].astype(np.int64)
    h2 = data[:, :, K + 1].astype(np.int64)
    ii = np.arange(data.shape[0])[:, None]
    tt = np.arange(args.sites)[None, :]
    true_match = feats[ii, tt, h1]
    switches = (h1[:, 1:] != h1[:, :-1]).sum(1)

    print(f"\nWrote {out_path}")
    print(f"  shape={data.shape}  dtype={data.dtype}  "
          f"size={data.nbytes / 1e9:.2f} GB")
    print(f"  mean allele sharing : {feats.mean():.4f}  (target {args.allele_sharing})")
    print(f"  true-founder match  : {true_match.mean():.4f}  "
          f"(expect ~{1 - args.bad_frac + args.bad_frac * (args.allele_sharing * K - 1) / (K - 1):.4f})")
    print(f"  crossovers/window   : mean={switches.mean():.2f}  "
          f"min={switches.min()}  max={switches.max()}")
    print(f"  H1==H2 fraction     : {(h1 == h2).all(1).mean():.4f}  "
          f"(target {args.inbreeding})")
    print(f"  label range         : [{data[:, :, K:].min()}, {data[:, :, K:].max()}]")


if __name__ == "__main__":
    main()
