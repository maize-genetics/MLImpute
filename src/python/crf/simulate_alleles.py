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
independent path.

Diploid read sampling: each of the T positions is ONE read drawn from a single
gamete — H1's chromosome with prob `--gamete-balance`, else H2's. The site's
feature is the allele-match pattern of that one gamete's founder only (not both).
So in an outbred window (F=0) consecutive reads alternate between two independent
founder paths, and the model must disentangle the interleaved sharing patterns
using continuity within each path. Hemizygosity/allele-dropout is intrinsic (only
one chromosome is seen per site); `--bad-frac` adds explicit corrupt/missing
reads. When F=1 (h1==h2) the active founder is identical for both gametes, so the
output is unchanged from the haploid layout (backward compatible).

Recombination-rate heterogeneity (E2): with --recomb-span > 1 the breakpoints are
placed proportional to a per-window, per-site recombination-rate map that is
piecewise-constant in `--recomb-tile`-site blocks and spans `recomb_span`-fold
(log-uniform hot/cold regions).  This rate map is HIDDEN — it is never a model input.
The encoder must *infer* the local recombination rate from the allele-match patterns
(LD structure) and feed it into the CRF transition cost c_t, because at inference the
true map is unknown.  The true rate is appended as a trailing column for EVALUATION
ONLY (to test whether the inferred c_t tracks it); the training dataset slices only
the K feature columns, so it is never seen during training.
With --recomb-span 1 (default) recombination is spatially uniform and no track is
emitted, reproducing the E1 layout.

Output  (matches LabeledDatasetDiploid):
    NPY int8 tensor [windows, sites, founders + 2 (+1)]
      cols 0:K   allele-match features
      col  K     H1 founder label  (0..K-1)
      col  K+1   H2 founder label  (0..K-1)
      col  K+2   per-site TRUE recomb rate  (eval-only diagnostic, never a model
                 input; present only when --recomb-span > 1)

Usage:
    pixi run -- python src/python/crf/simulate_alleles.py \
        --workdir /workdir/esb33 --windows 100000               # E1 (uniform)
    pixi run -- python src/python/crf/simulate_alleles.py \
        --workdir /workdir/esb33 --recomb-span 100 --out sim_alleles_e2.npy  # E2
"""

import argparse
from pathlib import Path

import numpy as np


def _rate_map(rng, n, T, span, tile):
    """Hidden per-window recomb-rate map [n, T] in [1, span].

    Piecewise-constant in `tile`-site blocks, each block log-uniform over
    [1, span] so the rate varies ~span-fold between hot and cold regions.
    """
    n_tiles = int(np.ceil(T / tile))
    coarse = np.exp(rng.uniform(0.0, np.log(span), size=(n, n_tiles)))
    return np.repeat(coarse, tile, axis=1)[:, :T]


def _build_paths(rng, n, T, K, n_cross, rate=None):
    """Piecewise-constant founder paths [n, T] with exactly n_cross switches each.

    n_cross is a [n] array; windows are grouped by switch count so the whole
    thing stays vectorized.  Adjacent segments are guaranteed to differ.
    If `rate` [n, T] is given, breakpoints are placed proportional to the local
    rate (Gumbel-top-k sampling without replacement); else uniform.
    """
    path = np.empty((n, T), dtype=np.int64)
    for v in np.unique(n_cross):
        rows = np.flatnonzero(n_cross == v)
        nv = rows.size

        # v distinct breakpoints in [1, T-1] per window
        if rate is None:
            bp = np.argsort(rng.random((nv, T - 1)), axis=1)[:, :v] + 1
        else:
            # Gumbel-top-k: top-v of (log rate + Gumbel) samples v distinct
            # positions with prob proportional to the local recomb rate.
            gum = -np.log(-np.log(rng.random((nv, T - 1))))
            keys = np.log(rate[rows][:, 1:]) + gum
            bp = np.argsort(-keys, axis=1)[:, :v] + 1
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


def _good_mask(rng, n, T, bad_frac, block):
    """Boolean [n, T] of 'good' (uncorrupted) sites.

    block <= 1  → independent per-site (P(bad)=bad_frac), identical to the
    original sim.  block > 1 → autocorrelated bad runs via a 2-state Markov
    chain with stationary P(bad)=bad_frac and mean bad-run length = block, so
    genotyping/nomination error arrives in tracts (SV blocks, mis-mapping).
    """
    if block <= 1.0:
        return rng.random((n, T)) >= bad_frac
    p_bg = 1.0 / block                                  # bad → good
    p_gb = bad_frac * p_bg / max(1e-9, 1.0 - bad_frac)  # good → bad (sets stationary)
    good = np.empty((n, T), dtype=bool)
    u = rng.random((n, T))
    good[:, 0] = u[:, 0] >= bad_frac
    for t in range(1, T):
        prev = good[:, t - 1]
        flip = np.where(prev, u[:, t] < p_gb, u[:, t] < p_bg)
        good[:, t] = np.where(flip, ~prev, prev)
    return good


def _coalescent_feats(rng, n, T, K, A, anc_cx, sfs_shape, read_snps,
                      h1, h2, rate, good, gamete):
    """Mini-haplotype (read) match features [n, T, K] from a coalescent panel.

    Each founder is a piecewise-constant mosaic over A ancestral lineages
    (built with the same rate map as the sample paths, so recombination
    hotspots shorten founder IBD tracts).  Founders copying the same ancestor
    are identical-by-descent → shared haplotype tracts (LD + relatedness).

    Each position is a 150bp read spanning `read_snps` (L) biallelic SNPs whose
    per-SNP derived-allele frequencies ~ Beta(sfs_shape, 1) (sfs_shape < 1 ⇒
    rare derived / many shared common alleles).  A founder "matches" a read
    only if its mini-haplotype agrees across ALL L SNPs (RopeBWT exact match),
    so multi-SNP reads are rare composites → a full-read match is a strong IBD
    signal, not a common-allele coincidence.  Returns feats[n, T, K].
    """
    L = read_snps
    anc_nc = rng.poisson(anc_cx, n * K).clip(0, T - 1).astype(np.int64)
    rate_rep = None if rate is None else np.repeat(rate, K, axis=0)
    anc_path = _build_paths(rng, n * K, T, A, anc_nc, rate_rep)
    anc_path = anc_path.reshape(n, K, T).astype(np.int32)

    f = rng.beta(sfs_shape, 1.0, size=(n, 1, T, L))     # per-SNP derived freq
    anc_alleles = (rng.random((n, A, T, L)) < f).astype(np.int8)   # [n,A,T,L]

    wi = np.arange(n)[:, None, None]
    ti = np.arange(T)[None, None, :]
    G = anc_alleles[wi, anc_path, ti]                   # [n,K,T,L] mini-haplotypes
    ii = np.arange(n)[:, None]
    tt = np.arange(T)[None, :]
    # One read per site, sampled from the active gamete (H1 if gamete else H2).
    active = np.where(gamete, h1, h2)                   # [n,T]
    Sa = G[ii, active, tt]                              # active mini-hap read [n,T,L]
    bad = ~good                                         # dropout/corrupt whole read in tracts
    fz = f[:, 0]                                        # [n,T,L]
    Sa = np.where(bad[:, :, None], (rng.random((n, T, L)) < fz).astype(np.int8), Sa)

    match = (G == Sa[:, None]).all(-1)                  # [n,K,T] exact full-read match
    return np.transpose(match, (0, 2, 1)).astype(np.int8)


def simulate(rng, windows, sites, founders, min_cross, max_cross,
             inbreeding, allele_sharing, bad_frac, recomb_span=1.0,
             recomb_tile=64, sharing_model="independent", ancestors=6,
             ancestor_crossovers=8, derived_sfs=0.3, read_snps=8,
             error_block=1.0, gamete_balance=0.5, chunk=1000):
    K = founders
    T = sites
    coalescent = sharing_model == "coalescent"
    if not coalescent:
        q = (allele_sharing * K - 1.0) / (K - 1)   # match rate for non-true founders
        if q < 0:
            raise ValueError(
                f"allele_sharing={allele_sharing} too low for K={K}; "
                f"minimum is {1.0 / K:.4f}")

    track = recomb_span > 1.0                       # emit hidden true-rate column
    ncol = K + 2 + (1 if track else 0)
    out = np.empty((windows, T, ncol), dtype=np.int8)

    for start in range(0, windows, chunk):
        n = min(chunk, windows - start)
        rmap = _rate_map(rng, n, T, recomb_span, recomb_tile) if track else None

        n_cross = rng.integers(min_cross, max_cross + 1, n)
        h1 = _build_paths(rng, n, T, K, n_cross, rmap)

        # Second haplotype: identical with prob F (inbred), else independent.
        # Outbred H2 shares the same hidden rate map (same genomic region).
        inbred = rng.random(n) < inbreeding
        h2 = h1.copy()
        outbred = np.flatnonzero(~inbred)
        if outbred.size:
            h2[outbred] = _build_paths(
                rng, outbred.size, T, K,
                rng.integers(min_cross, max_cross + 1, outbred.size),
                None if rmap is None else rmap[outbred])

        good = _good_mask(rng, n, T, bad_frac, error_block)

        # Per-site read origin: True => sampled from H1's chromosome, else H2's.
        # gamete_balance = P(H1). One read per site (diploid low-coverage sampling).
        gamete = rng.random((n, T)) < gamete_balance
        active = np.where(gamete, h1, h2)               # observed founder per site

        if coalescent:
            feats = _coalescent_feats(
                rng, n, T, K, ancestors, ancestor_crossovers,
                derived_sfs, read_snps, h1, h2, rmap, good, gamete)
        else:
            # Independent background; force only the ACTIVE gamete's founder to
            # match on good sites (one read, from one chromosome).
            feats = (rng.random((n, T, K)) < q).astype(np.int8)
            ii = np.arange(n)[:, None]
            tt = np.arange(T)[None, :]
            feats[ii, tt, active] = np.where(good, 1, feats[ii, tt, active])

        out[start:start + n, :, :K] = feats
        out[start:start + n, :, K] = h1.astype(np.int8)
        out[start:start + n, :, K + 1] = h2.astype(np.int8)
        if track:
            out[start:start + n, :, K + 2] = np.clip(
                np.rint(rmap), 1, 127).astype(np.int8)

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
                   help="Inbreeding coefficient F in [0,1]; P(H1==H2 path). "
                        "F=0 → fully outbred diploid (interleaved single-gamete reads)")
    p.add_argument("--gamete-balance", type=float, default=0.5,
                   help="P(a site's read is sampled from H1's chromosome); 0.5 = even. "
                        "Skew simulates one chromosome dominating coverage.")
    p.add_argument("--allele-sharing", type=float, default=0.2,
                   help="Average fraction of founders sharing the sample allele")
    p.add_argument("--sharing-model", choices=["independent", "coalescent"],
                   default="independent",
                   help="independent: per-site random sharing (default). coalescent: "
                        "mosaic-of-ancestors panel → LD tracts + founder relatedness.")
    p.add_argument("--ancestors", type=int, default=6,
                   help="Coalescent mode: number of ancestral lineages A")
    p.add_argument("--ancestor-crossovers", type=int, default=8,
                   help="Coalescent mode: mean ancestor switches per founder per window")
    p.add_argument("--derived-sfs", type=float, default=0.3,
                   help="Coalescent mode: Beta(shape,1) site-frequency-spectrum shape; "
                        "<1 ⇒ rare derived alleles / many shared common alleles")
    p.add_argument("--read-snps", type=int, default=8,
                   help="Coalescent mode: SNPs per mini-haplotype read (exact-match "
                        "length L); more SNPs ⇒ rarer, more-specific full-read matches")
    p.add_argument("--bad-frac", type=float, default=0.05,
                   help="Proportion of sites with corrupted (random) patterns")
    p.add_argument("--error-block", type=float, default=1.0,
                   help="Mean length (sites) of correlated error runs; 1 = independent")
    p.add_argument("--recomb-span", type=float, default=1.0,
                   help="Hot/cold recomb-rate ratio (E2). >1 places breakpoints by a "
                        "hidden variable rate map + appends the true rate as an "
                        "eval-only column. 1 = uniform (E1 layout, no track).")
    p.add_argument("--recomb-tile", type=int, default=64,
                   help="Block size (sites) of constant recomb rate")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data = simulate(
        rng, args.windows, args.sites, args.founders,
        args.min_crossovers, args.max_crossovers,
        args.inbreeding, args.allele_sharing, args.bad_frac,
        args.recomb_span, args.recomb_tile,
        args.sharing_model, args.ancestors, args.ancestor_crossovers,
        args.derived_sfs, args.read_snps, args.error_block, args.gamete_balance)

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
    m1 = feats[ii, tt, h1]                       # H1 founder match indicator
    m2 = feats[ii, tt, h2]                       # H2 founder match indicator
    either = np.maximum(m1, m2)                  # active gamete is one of the two
    het = (h1 != h2)                             # heterozygous sites
    switches = (h1[:, 1:] != h1[:, :-1]).sum(1)

    print(f"\nWrote {out_path}")
    print(f"  shape={data.shape}  dtype={data.dtype}  "
          f"size={data.nbytes / 1e9:.2f} GB")
    print(f"  mean allele sharing : {feats.mean():.4f}  (target {args.allele_sharing})")
    print(f"  either-founder match: {either.mean():.4f}  "
          f"(active gamete forced on good sites; expect ~{1 - args.bad_frac:.3f}+)")
    print(f"  H1 / H2 match (het) : {m1[het].mean():.4f} / {m2[het].mean():.4f}  "
          f"(gamete-balance {args.gamete_balance}; only the sampled gamete is forced)")
    print(f"  het-site fraction   : {het.mean():.4f}  "
          f"(F={args.inbreeding}; 0 ⇒ mostly het, interleaved reads)")
    print(f"  crossovers/window   : mean={switches.mean():.2f}  "
          f"min={switches.min()}  max={switches.max()}")
    print(f"  H1==H2 fraction     : {(h1 == h2).all(1).mean():.4f}  "
          f"(target {args.inbreeding})")
    print(f"  label range         : [{data[:, :, K].min()}, {data[:, :, K + 1].max()}]")

    if data.shape[-1] == K + 3:
        rate = data[:, :, K + 2].astype(np.float64)
        sw = np.zeros_like(rate)
        sw[:, 1:] = (h1[:, 1:] != h1[:, :-1])      # switch entering site t
        r = np.corrcoef(rate.ravel(), sw.ravel())[0, 1]
        print(f"  recomb rate (hidden): min={rate.min():.0f}  max={rate.max():.0f}  "
              f"mean={rate.mean():.1f}  (span target {args.recomb_span:.0f})")
        print(f"  corr(rate, switch)  : {r:.4f}  "
              f"(positive ⇒ breakpoints follow the hidden rate; eval-only column)")

    # Feature LD: lag-1 autocorrelation of the per-founder match indicator.
    # Independent sharing ≈ 0; coalescent > 0 (shared IBD tracts).  When a rate
    # track is present, LD should be weaker in hotspots (faster tract breakdown).
    def _lag1(fa):
        a = fa[:, 1:, :].astype(np.float64)
        b = fa[:, :-1, :].astype(np.float64)
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            return 0.0
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    print(f"  match LD (lag-1 ac) : {_lag1(feats):.4f}  "
          f"(model={args.sharing_model}; >0 ⇒ allele-sharing tracts)")
    if data.shape[-1] == K + 3:
        rate_t = data[:, :, K + 2]
        hot = rate_t[:, :-1] >= np.percentile(rate_t, 90)
        cold = rate_t[:, :-1] <= np.percentile(rate_t, 10)
        same = (feats[:, 1:, :] == feats[:, :-1, :])
        ac_hot = same[hot].mean() if hot.any() else float("nan")
        ac_cold = same[cold].mean() if cold.any() else float("nan")
        print(f"  match persist hot/cold: {ac_hot:.4f} / {ac_cold:.4f}  "
              f"(cold > hot ⇒ recombination is observable in the features)")


if __name__ == "__main__":
    main()
