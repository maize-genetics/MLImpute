#!/usr/bin/env python
"""
Recombination-simulation Tripsacum alignment test: extends tripsacum_diploid.py's simple
(no-recombination) synthetic diploid to genuinely RECOMBINING individuals -- H1 and H2 each
independently switch between the same two founders at N simulated crossover breakpoints per
chromosome, for N in {1, 2, 5, 10}, swept across the same 4 founder pairs used in the simple
diploid test AND across 5 random seeds per (pair, level) -- averaged, to separate a real
breakpoint-density effect from single-draw het_frac noise (an earlier single-seed run showed
a large, solid no-recombination-vs-any-recombination gap but a non-monotonic 1/2/5/10 trend
that turned out to be confounded with each level's one random draw's realized het_frac).

Why this doesn't splice literal recombinant FASTA sequence: only the 4 PanAnd reference
assemblies are chromosome-scale; the 14 non-reference assemblies (including all 4 founder
pairs used here) are raw hifiasm contigs (thousands of unordered/unoriented contigs each),
not position-addressable at the chromosome level. Instead, this works entirely in the
pangenome's own reference-projected coordinate space (the same refmap/lift machinery
already validated):

  1. Reuse the UNCHANGED whole-genome read simulation + refmap --npy export + K=18
     windowing from the prior tripsacum_diploid.py run for each pair/depth (cached, not
     re-run) -- all 4 pairs' base runs already exist.
  2. Independently sample crossover breakpoints along each of the 18 reference chromosomes
     for H1 and H2 (build_mosaic_map), giving each haplotype its own piecewise-constant
     A/B ancestry track. A distinct, documented seed per (pair, level, seed_idx).
  3. Reconstruct each windowed site's approximate reference bp position by replicating
     ropebwt_npy_to_matrix.py's exact row-grouping logic against raw.npy.bins.tsv
     (site_positions) -- no changes to the windowing script itself.
  4. For every site, mask out (zero) any founder's feature column that isn't part of the
     true local (H1,H2) ancestry pair -- this is what a real recombinant individual's reads
     would actually look like (no evidence for an absent parent in that segment) -- and
     write the true, now per-site-varying, (H1,H2) labels (assign_and_mask).
  5. Pad K=18->24 (tripsacum_diploid.pad_to_24, unchanged) and evaluate
     (tripsacum_diploid.run_diploid_eval, unchanged).

Verified this is valid at full per-site resolution, not just per-window: read
train_diploid.py's PreWindowedDiploidDataset.__getitem__ directly -- h1/h2 are per-site
[T]-length label vectors, and evaluate_diploid already scores accuracy per site with no
assumption of label constancy across a window. A breakpoint landing inside a 512-row
window is therefore fully representable and scoreable. Also verified (single-seed run):
mosaic-map alternation, position-alignment against bins.tsv, and masking correctness on
real homozygous/heterozygous windows -- all passed.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py one <A> <B> <n_breakpoints> <seed_idx>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py all
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py report
    # affinity-model variants (same args, checkpoints/diploid-affinity-sim512-h3, via
    # tripsacum_diploid.run_diploid_eval's affinity auto-detection):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py one-affinity <A> <B> <n_breakpoints> <seed_idx>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py all-affinity
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination.py report-affinity
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tripsacum_diploid as td  # noqa: E402  (reuse PAIRS/BIN/FMD/LIFT/CHR_LENGTHS/
# combine_reads/run_refmap_combined/discover_panel_order/pad_to_24/run_diploid_eval)

DEPTH = 250_000
LEVELS = [1, 2, 5, 10]  # breakpoints per chromosome
N_SEEDS = 5             # seeds averaged per (pair, level)
WINDOW_SIZE = 512
BIN_SIZE = 256  # refmap --npy default bin size in bp (unchanged from the existing runs)

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/tripsacum_recombination")
DETAIL_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_recombination_detail.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_recombination.md")
AFFINITY_DETAIL_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "tripsacum_recombination_affinity_detail.tsv")
AFFINITY_RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "tripsacum_recombination_affinity.md")


def seed_for(n_breakpoints, seed_idx):
    """Deterministic, reproducible: distinct per (level, seed_idx). Independent of pair --
    the pair only determines which two founder columns are used, not the breakpoint/founder-
    start draw itself, so the same geometry is reused across pairs (fine: the two founders'
    *roles* as 'A'/'B' are arbitrary labels resolved to real panel indices downstream)."""
    return 1000 + n_breakpoints * 100 + seed_idx


def base_run_dir(a, b):
    """The already-completed tripsacum_diploid.py run for this pair/depth -- reused
    unchanged for reads/refmap/windowing (only the mosaic/mask/label step differs per
    recombination level/seed)."""
    name = td.combo_name([a, b], DEPTH)
    d = td.SCRATCH / name
    for f in ("windowed_k18.npy", "raw.npy.bins.tsv", "raw.npy.gametes.tsv"):
        if not (d / f).exists():
            raise RuntimeError(
                f"{d / f} missing -- run `tripsacum_diploid.py one {a} {b} {DEPTH}` first "
                f"(this script reuses its reads/refmap/windowing output).")
    return d


def build_mosaic_map(chr_lengths, n_breakpoints, seed):
    """Per chromosome, per haplotype (H1, H2) independently: n_breakpoints sorted uniform
    bp positions, alternating A/B founder segments (starting founder randomized). Returns
    {(chrom, hap): {"breakpoints": sorted int array, "founders": list of 'A'/'B',
    length n_breakpoints+1}}."""
    rng = np.random.default_rng(seed)
    mosaic = {}
    for chrom, length in chr_lengths:
        for hap in ("H1", "H2"):
            bps = np.sort(rng.integers(1, length, size=n_breakpoints)) if n_breakpoints else \
                np.array([], dtype=np.int64)
            start = "A" if rng.random() < 0.5 else "B"
            founders = []
            f = start
            for _ in range(n_breakpoints + 1):
                founders.append(f)
                f = "B" if f == "A" else "A"
            mosaic[(chrom, hap)] = {"breakpoints": bps, "founders": founders, "length": length}
    return mosaic


def lookup_founders(mosaic, chrom, hap, positions):
    """Vectorized: for an array of bp positions on (chrom, hap), return the array of
    'A'/'B' founder letters active at each position."""
    m = mosaic[(chrom, hap)]
    seg_idx = np.searchsorted(m["breakpoints"], positions, side="right")
    return np.asarray(m["founders"])[seg_idx]


def site_positions(bins_tsv_path, window_size=WINDOW_SIZE):
    """Reimplements ropebwt_npy_to_matrix.py's exact row grouping (group by contig, sort
    by bin, chunk into non-overlapping window_size-row windows, drop trailing partial
    window per contig) to produce [N_windows, window_size] arrays of (contig, bp) aligned
    row-for-row with windowed_k18.npy's own layout. bp = bin * BIN_SIZE."""
    bins_df = pd.read_csv(bins_tsv_path, sep="\t")
    assert list(bins_df.columns) == ["row", "contig", "bin"], \
        f"unexpected bins.tsv columns: {list(bins_df.columns)}"
    assert np.array_equal(bins_df["row"].to_numpy(), np.arange(len(bins_df))), \
        "bins.tsv 'row' column is not a plain 0..N-1 sequence"

    contig_col = bins_df["contig"].to_numpy()
    bin_col = bins_df["bin"].to_numpy()

    contigs_out, bps_out = [], []
    for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            sel = idx[start:start + window_size]
            contigs_out.append(contig_col[sel])
            bps_out.append(bin_col[sel] * BIN_SIZE)
            start += window_size

    if not contigs_out:
        raise ValueError("no complete windows -- window_size too large for the data")
    contigs = np.stack(contigs_out, axis=0)
    bps = np.stack(bps_out, axis=0)
    return contigs, bps


def assign_and_mask(windowed_k18_path, contigs, bps, mosaic, panel, a, b, out_path):
    """For every site: mask (zero) every founder feature column that isn't part of the true
    local (H1, H2) ancestry pair, and write the true per-site labels. Sites on a contig with
    no defined mosaic (e.g. non-canonical reference scaffolds outside chr1..chr18) are left
    as the 'unknown' state (index K18), matching ropebwt_npy_to_matrix.py's own -1->K
    unknown-label convention."""
    if out_path.exists():
        return out_path

    arr = np.load(windowed_k18_path).copy()
    N, T, C = arr.shape
    K18 = C - 2
    assert (N, T) == contigs.shape == bps.shape, (
        f"site_positions shape {contigs.shape} != windowed array shape {(N, T)} -- "
        f"row-grouping replication mismatch")

    idxA, idxB = panel[a], panel[b]

    h1_idx = np.full((N, T), K18, dtype=np.int64)  # default: unknown
    h2_idx = np.full((N, T), K18, dtype=np.int64)
    keep_mask = np.zeros((N, T, K18), dtype=bool)

    for chrom in np.unique(contigs):
        if (chrom, "H1") not in mosaic:
            continue  # no mosaic defined for this contig (e.g. unplaced scaffold) -- unknown
        sel = contigs == chrom
        rows, cols = np.where(sel)
        pos = bps[sel]

        f1 = lookup_founders(mosaic, chrom, "H1", pos)
        f2 = lookup_founders(mosaic, chrom, "H2", pos)
        f1_idx = np.where(f1 == "A", idxA, idxB)
        f2_idx = np.where(f2 == "A", idxA, idxB)

        h1_idx[sel] = f1_idx
        h2_idx[sel] = f2_idx
        keep_mask[rows, cols, f1_idx] = True
        keep_mask[rows, cols, f2_idx] = True

    feats = arr[:, :, :K18]
    arr[:, :, :K18] = np.where(keep_mask, feats, 0)
    arr[:, :, K18] = h1_idx
    arr[:, :, K18 + 1] = h2_idx

    np.save(out_path, arr)
    return out_path


DETAIL_COLS = ["assemblyA", "assemblyB", "n_breakpoints_per_chrom", "seed_idx", "seed",
               "het_frac", "unknown_frac", "n_sites", "pair_acc", "hap_acc", "homo_pred"]


def write_header_if_needed(detail_tsv=DETAIL_TSV):
    detail_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not detail_tsv.exists():
        detail_tsv.write_text("\t".join(DETAIL_COLS) + "\n")


def already_recorded(a, b, n_breakpoints, seed_idx, detail_tsv=DETAIL_TSV):
    if not detail_tsv.exists():
        return False
    key = f"{a}\t{b}\t{n_breakpoints}\t{seed_idx}\t"
    with open(detail_tsv) as f:
        return any(line.startswith(key) for line in f)


def run_one(a, b, n_breakpoints, seed_idx, device, force=False,
            ckpt_path=td.DIPLOID_CKPT, detail_tsv=DETAIL_TSV):
    if not force and already_recorded(a, b, n_breakpoints, seed_idx, detail_tsv):
        print(f"[{a}x{b} n={n_breakpoints} seed_idx={seed_idx}] already recorded, skipping")
        return

    seed = seed_for(n_breakpoints, seed_idx)
    name = f"{a}x{b}_recomb_{n_breakpoints}bp_seed{seed_idx}"
    print(f"\n=== {name} (seed={seed}) ===")
    base_dir = base_run_dir(a, b)
    outdir = SCRATCH / name
    outdir.mkdir(parents=True, exist_ok=True)

    panel = td.discover_panel_order(base_dir / "raw.npy.gametes.tsv")
    mosaic = build_mosaic_map(td.CHR_LENGTHS, n_breakpoints, seed=seed)
    contigs, bps = site_positions(base_dir / "raw.npy.bins.tsv")

    masked_k18 = assign_and_mask(base_dir / "windowed_k18.npy", contigs, bps, mosaic, panel,
                                  a, b, outdir / "masked_k18.npy")
    padded = td.pad_to_24(masked_k18, outdir)

    arr = np.load(padded, mmap_mode="r")
    K = arr.shape[-1] - 2
    het_frac = float((arr[:, :, K] != arr[:, :, K + 1]).mean())
    unknown_frac = float(((arr[:, :, K] == K) | (arr[:, :, K + 1] == K)).mean())

    r = td.run_diploid_eval(padded, name, device, ckpt_path=ckpt_path)
    row = dict(assemblyA=a, assemblyB=b, n_breakpoints_per_chrom=n_breakpoints,
               seed_idx=seed_idx, seed=seed, het_frac=het_frac, unknown_frac=unknown_frac,
               n_sites=r["n"], pair_acc=r["pair_acc"], hap_acc=r["hap_acc"],
               homo_pred=r["homo_pred"])

    write_header_if_needed(detail_tsv)
    if force and already_recorded(a, b, n_breakpoints, seed_idx, detail_tsv):
        lines = detail_tsv.read_text().splitlines(keepends=True)
        key = f"{a}\t{b}\t{n_breakpoints}\t{seed_idx}\t"
        keep = [l for l in lines if not l.startswith(key)]
        detail_tsv.write_text("".join(keep))
    with open(detail_tsv, "a") as f:
        f.write("\t".join(str(row[c]) for c in DETAIL_COLS) + "\n")
    print(f"[{name}] het_frac={het_frac:.4f}  unknown_frac={unknown_frac:.4f}  "
          f"pair_acc={row['pair_acc']:.4f}  hap_acc={row['hap_acc']:.4f}  "
          f"homo_pred={row['homo_pred']:.4f}")


def _markdown_table(df, float_cols):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if (c in float_cols and isinstance(v, float)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report(detail_tsv=DETAIL_TSV, results_md=RESULTS_MD, affinity=False):
    detail = pd.read_csv(detail_tsv, sep="\t")

    agg = (detail.groupby(["assemblyA", "assemblyB", "n_breakpoints_per_chrom"])
           .agg(n_seeds=("seed_idx", "count"),
                het_frac_mean=("het_frac", "mean"), het_frac_std=("het_frac", "std"),
                pair_acc_mean=("pair_acc", "mean"), pair_acc_std=("pair_acc", "std"),
                hap_acc_mean=("hap_acc", "mean"), hap_acc_std=("hap_acc", "std"),
                homo_pred_mean=("homo_pred", "mean"), homo_pred_std=("homo_pred", "std"))
           .reset_index()
           .sort_values(["assemblyA", "assemblyB", "n_breakpoints_per_chrom"]))
    float_cols = {c for c in agg.columns if c.endswith(("_mean", "_std"))}
    per_pair_md = _markdown_table(agg, float_cols)

    overall = (detail.groupby("n_breakpoints_per_chrom")
               .agg(n_runs=("seed_idx", "count"),
                    het_frac_mean=("het_frac", "mean"),
                    pair_acc_mean=("pair_acc", "mean"), pair_acc_std=("pair_acc", "std"),
                    hap_acc_mean=("hap_acc", "mean"))
               .reset_index()
               .sort_values("n_breakpoints_per_chrom"))
    overall_float_cols = {c for c in overall.columns if c.endswith(("_mean", "_std"))}
    overall_md = _markdown_table(overall, overall_float_cols)

    model_desc = ("the **affinity-conditioned** diploid GRITS-CRF "
                   "(`checkpoints/diploid-affinity-sim512-h3`), fed a genome-wide "
                   "`_founder_affinity` vector computed from each run's own data "
                   "(`tripsacum_diploid.evaluate_diploid_with_affinity`)"
                   if affinity else
                   "the plain (non-affinity) diploid GRITS-CRF via "
                   "`crf/eval.py::evaluate_diploid`")
    lines = [
        f"# Recombination-simulation Tripsacum baseline "
        f"({'affinity' if affinity else 'plain'} diploid CRF) -- "
        f"{N_SEEDS} seeds x 4 pairs, averaged\n",
        "H1 and H2 each independently recombine between two founders at N breakpoints per "
        "chromosome (uniform random positions, alternating founder per segment), instead "
        "of being constant across the whole genome (the earlier no-recombination baseline, "
        "`results/tripsacum_diploid.md`). Reads/refmap/K=18-windowing are reused unchanged "
        "per pair from the simple diploid runs; per (pair, level, seed), founder feature "
        "columns are masked to only the locally-true ancestry pair and the true per-site "
        "`(H1,H2)` labels are written before padding K=18->24 (the checkpoint is hard-fixed "
        f"to 24 founders) and scoring with {model_desc}. "
        f"{N_SEEDS} independent random breakpoint/founder-assignment draws are averaged per "
        "(pair, level) to separate a real breakpoint-density effect from single-draw "
        "het_frac noise (an earlier single-seed run's non-monotonic 1/2/5/10 trend turned "
        "out to be confounded this way).\n\n"
        "## Overall (averaged across all 4 pairs and all seeds)\n\n"
        f"{overall_md}\n\n"
        "## Per pair (mean +/- std across seeds)\n\n"
        f"{per_pair_md}\n\n"
        f"Full per-seed detail: `{detail_tsv.name}`.\n",
    ]
    results_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {results_md}")
    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nPer pair:")
    print(agg.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]

    if mode == "list":
        print(f"Pairs: {td.PAIRS}")
        print(f"Depth: {DEPTH:,}/hap")
        print(f"Levels (breakpoints/chrom): {LEVELS}")
        print(f"Seeds per (pair, level): {N_SEEDS}")
        for a, b in td.PAIRS:
            try:
                base_run_dir(a, b)
                print(f"  {a} x {b}: base diploid run OK")
            except RuntimeError as e:
                print(f"  {a} x {b}: MISSING -- {e}")
        print(f"\nTotal runs: {len(td.PAIRS)} pairs x {len(LEVELS)} levels x "
              f"{N_SEEDS} seeds = {len(td.PAIRS) * len(LEVELS) * N_SEEDS}")
        return

    if mode == "report":
        write_report()
        return

    if mode == "report-affinity":
        write_report(AFFINITY_DETAIL_TSV, AFFINITY_RESULTS_MD, affinity=True)
        return

    if mode in ("one", "one-affinity"):
        if len(sys.argv) < 6:
            raise SystemExit(f"usage: tripsacum_recombination.py {mode} <A> <B> <n_breakpoints> <seed_idx>")
        a, b, n, seed_idx = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = td.AFFINITY_CKPT if mode == "one-affinity" else td.DIPLOID_CKPT
        tsv = AFFINITY_DETAIL_TSV if mode == "one-affinity" else DETAIL_TSV
        write_header_if_needed(tsv)
        run_one(a, b, n, seed_idx, device, force=True, ckpt_path=ckpt, detail_tsv=tsv)
    elif mode in ("all", "all-affinity"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = td.AFFINITY_CKPT if mode == "all-affinity" else td.DIPLOID_CKPT
        tsv = AFFINITY_DETAIL_TSV if mode == "all-affinity" else DETAIL_TSV
        write_header_if_needed(tsv)
        for a, b in td.PAIRS:
            for n in LEVELS:
                for seed_idx in range(N_SEEDS):
                    run_one(a, b, n, seed_idx, device, ckpt_path=ckpt, detail_tsv=tsv)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {AFFINITY_DETAIL_TSV if 'affinity' in mode else DETAIL_TSV}")


if __name__ == "__main__":
    main()
