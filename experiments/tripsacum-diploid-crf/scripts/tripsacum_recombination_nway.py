#!/usr/bin/env python
"""
N-way recombination-simulation Tripsacum alignment test: generalizes
tripsacum_recombination.py from exactly 2 founders (strict A/B alternation) to a pool of
N founders (4 or 6) -- a mosaic closer to a real multi-parent/MAGIC-style breeding
population, where H1/H2 can switch to ANY of the N founders at each breakpoint, not just
flip between two fixed roles.

Founder sets (reusing this session's own work, not arbitrary new assemblies):
  N=4: the "hub" cluster already central to every prior pair test
       (C009-T009, C011-T007, C027-T007, C050-T007)
  N=6: that same set plus the two individuals from the fourth (independent-member) pair
       (+ C076-T198, C081-T199)
  -- every distinct individual used anywhere in tripsacum_diploid.py's pair tests.

Mechanics, mirroring tripsacum_recombination.py exactly except where N>2 requires a real
change:
  1. Reads are simulated from all N founders and combined into one FASTQ, then run through
     ropebwt3 refmap + K=18 windowing ONCE per founder set (base_run()) -- there's no
     pre-existing tripsacum_diploid.py base run for N>2 founders to reuse, unlike the 2-way
     script, so this script builds its own (reusing the now-generalized
     tripsacum_diploid.combine_reads/combo_name, which accept a list of founders).
  2. Per chromosome, per haplotype (H1, H2) independently: each segment's founder is drawn
     uniformly at random from the N-pool, constrained to differ from the immediately
     preceding segment (build_mosaic_map) -- this generalizes tripsacum_recombination.py's
     strict alternation, which is just this same rule specialized to N=2 (only one other
     choice exists there, so "differ from previous" *is* alternation).
  3. Founder feature columns are masked to the locally-true (H1, H2) pair at each site
     (assign_and_mask, generalized to take the founder list instead of hardcoded a, b);
     true per-site labels; pad K=18->24; score. All unchanged from the 2-way script.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py one-nway <n_founders> <n_breakpoints> <seed_idx>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py all-nway
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py report-nway
    # affinity-model variants (same args, checkpoints/diploid-affinity-sim512-h3):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py one-nway-affinity <n_founders> <n_breakpoints> <seed_idx>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py all-nway-affinity
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_recombination_nway.py report-nway-affinity
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tripsacum_diploid as td  # noqa: E402  (reuse BIN/FMD/LIFT/CHR_LENGTHS/combine_reads/
# combo_name/run_refmap_combined/discover_panel_order/pad_to_24/run_diploid_eval)

DEPTH = 250_000
LEVELS = [1, 2, 5, 10]  # breakpoints per chromosome
N_SEEDS = 5             # seeds averaged per (founder-set, level)
WINDOW_SIZE = 512
BIN_SIZE = 256  # refmap --npy default bin size in bp (unchanged from the existing runs)

FOUNDER_SETS = {
    4: ["C009-T009", "C011-T007", "C027-T007", "C050-T007"],
    6: ["C009-T009", "C011-T007", "C027-T007", "C050-T007", "C076-T198", "C081-T199"],
}

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/tripsacum_recombination_nway")
DETAIL_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_recombination_nway_detail.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_recombination_nway.md")
AFFINITY_DETAIL_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "tripsacum_recombination_nway_affinity_detail.tsv")
AFFINITY_RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "tripsacum_recombination_nway_affinity.md")


def seed_for(n_founders, n_breakpoints, seed_idx):
    """Deterministic, reproducible: distinct per (n_founders, level, seed_idx)."""
    return 2000 + n_founders * 10000 + n_breakpoints * 100 + seed_idx


def base_run(founders, depth):
    """Build (or reuse if already cached) the combined-FASTQ base run for this founder
    set: reads -> refmap --npy -> K=18 windowing. Unlike tripsacum_recombination.py, there
    is no pre-existing tripsacum_diploid.py run to require/reuse for N>2 founders, so this
    builds it directly via the now-generalized td.combine_reads/td.combo_name (accept a
    list of founders) and td.run_refmap_combined/td.window (already founder-count-agnostic).
    Returns the base run's outdir."""
    name = td.combo_name(founders, depth)
    outdir = td.SCRATCH.parent / "tripsacum_recombination_nway_base" / name
    outdir.mkdir(parents=True, exist_ok=True)

    combined = td.combine_reads(founders, depth, outdir)
    raw_npy = td.run_refmap_combined(combined, outdir)
    td.window(raw_npy, outdir)  # writes windowed_k18.npy into outdir; also need bins/gametes there
    return outdir


def build_mosaic_map(chr_lengths, founders, n_breakpoints, seed):
    """Per chromosome, per haplotype (H1, H2) independently: n_breakpoints sorted uniform
    bp positions, each resulting segment's founder drawn uniformly from `founders`,
    constrained to differ from the immediately preceding segment (generalizes
    tripsacum_recombination.py's strict A/B alternation -- with only 2 founders,
    "differ from previous" has exactly one choice, i.e. is alternation). Returns
    {(chrom, hap): {"breakpoints": sorted int array, "founders": list of founder names,
    length n_breakpoints+1}}."""
    rng = np.random.default_rng(seed)
    n = len(founders)
    mosaic = {}
    for chrom, length in chr_lengths:
        for hap in ("H1", "H2"):
            bps = np.sort(rng.integers(1, length, size=n_breakpoints)) if n_breakpoints else \
                np.array([], dtype=np.int64)
            seg_founders = [founders[rng.integers(0, n)]]
            for _ in range(n_breakpoints):
                prev = seg_founders[-1]
                choices = [f for f in founders if f != prev]
                seg_founders.append(choices[rng.integers(0, len(choices))])
            mosaic[(chrom, hap)] = {"breakpoints": bps, "founders": seg_founders, "length": length}
    return mosaic


def lookup_founders(mosaic, chrom, hap, positions):
    """Vectorized: for an array of bp positions on (chrom, hap), return the array of
    founder names active at each position."""
    m = mosaic[(chrom, hap)]
    seg_idx = np.searchsorted(m["breakpoints"], positions, side="right")
    return np.asarray(m["founders"])[seg_idx]


def site_positions(bins_tsv_path, window_size=WINDOW_SIZE):
    """Identical to tripsacum_recombination.py's site_positions -- reimplements
    ropebwt_npy_to_matrix.py's exact row grouping to produce [N_windows, window_size]
    arrays of (contig, bp) aligned row-for-row with windowed_k18.npy's own layout."""
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


def assign_and_mask(windowed_k18_path, contigs, bps, mosaic, panel, founders, out_path):
    """For every site: mask (zero) every founder feature column that isn't part of the true
    local (H1, H2) ancestry pair, and write the true per-site labels. `founders`: the list
    of sample names in this founder set (used only to validate they're all in `panel`;
    lookup_founders already resolves real names directly, no letter-indirection needed for
    N>2). Sites on a contig with no defined mosaic are left as the 'unknown' state
    (index K18), matching ropebwt_npy_to_matrix.py's own -1->K unknown-label convention."""
    if out_path.exists():
        return out_path

    arr = np.load(windowed_k18_path).copy()
    N, T, C = arr.shape
    K18 = C - 2
    assert (N, T) == contigs.shape == bps.shape, (
        f"site_positions shape {contigs.shape} != windowed array shape {(N, T)} -- "
        f"row-grouping replication mismatch")

    founder_idx = {f: panel[f] for f in founders}

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
        f1_idx = np.vectorize(founder_idx.get)(f1)
        f2_idx = np.vectorize(founder_idx.get)(f2)

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


DETAIL_COLS = ["n_founders", "founder_set", "n_breakpoints_per_chrom", "seed_idx", "seed",
               "het_frac", "unknown_frac", "n_sites", "pair_acc", "hap_acc", "homo_pred"]


def write_header_if_needed(detail_tsv=DETAIL_TSV):
    detail_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not detail_tsv.exists():
        detail_tsv.write_text("\t".join(DETAIL_COLS) + "\n")


def already_recorded(n_founders, n_breakpoints, seed_idx, detail_tsv=DETAIL_TSV):
    if not detail_tsv.exists():
        return False
    key = f"{n_founders}\t"
    key2 = f"\t{n_breakpoints}\t{seed_idx}\t"
    with open(detail_tsv) as f:
        return any(line.startswith(key) and key2 in line for line in f)


def run_one(n_founders, n_breakpoints, seed_idx, device, force=False,
            ckpt_path=td.DIPLOID_CKPT, detail_tsv=DETAIL_TSV):
    if n_founders not in FOUNDER_SETS:
        raise SystemExit(f"n_founders must be one of {sorted(FOUNDER_SETS)}, got {n_founders}")
    if not force and already_recorded(n_founders, n_breakpoints, seed_idx, detail_tsv):
        print(f"[n={n_founders} bp={n_breakpoints} seed_idx={seed_idx}] already recorded, skipping")
        return

    founders = FOUNDER_SETS[n_founders]
    seed = seed_for(n_founders, n_breakpoints, seed_idx)
    name = f"nway{n_founders}_recomb_{n_breakpoints}bp_seed{seed_idx}"
    print(f"\n=== {name} ({'+'.join(founders)}, seed={seed}) ===")

    base_dir = base_run(founders, DEPTH)
    outdir = SCRATCH / name
    outdir.mkdir(parents=True, exist_ok=True)

    panel = td.discover_panel_order(base_dir / "raw.npy.gametes.tsv")
    for f in founders:
        if f not in panel:
            raise RuntimeError(f"{name}: founder {f!r} not found in discovered panel {sorted(panel)}")
    mosaic = build_mosaic_map(td.CHR_LENGTHS, founders, n_breakpoints, seed=seed)
    contigs, bps = site_positions(base_dir / "raw.npy.bins.tsv")

    masked_k18 = assign_and_mask(base_dir / "windowed_k18.npy", contigs, bps, mosaic, panel,
                                  founders, outdir / "masked_k18.npy")
    padded = td.pad_to_24(masked_k18, outdir)

    arr = np.load(padded, mmap_mode="r")
    K = arr.shape[-1] - 2
    het_frac = float((arr[:, :, K] != arr[:, :, K + 1]).mean())
    unknown_frac = float(((arr[:, :, K] == K) | (arr[:, :, K + 1] == K)).mean())

    r = td.run_diploid_eval(padded, name, device, ckpt_path=ckpt_path)
    row = dict(n_founders=n_founders, founder_set="+".join(founders),
               n_breakpoints_per_chrom=n_breakpoints, seed_idx=seed_idx, seed=seed,
               het_frac=het_frac, unknown_frac=unknown_frac, n_sites=r["n"],
               pair_acc=r["pair_acc"], hap_acc=r["hap_acc"], homo_pred=r["homo_pred"])

    write_header_if_needed(detail_tsv)
    if force and already_recorded(n_founders, n_breakpoints, seed_idx, detail_tsv):
        lines = detail_tsv.read_text().splitlines(keepends=True)
        key = f"{n_founders}\t"
        key2 = f"\t{n_breakpoints}\t{seed_idx}\t"
        keep = [l for l in lines if not (l.startswith(key) and key2 in l)]
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

    agg = (detail.groupby(["n_founders", "founder_set", "n_breakpoints_per_chrom"])
           .agg(n_seeds=("seed_idx", "count"),
                het_frac_mean=("het_frac", "mean"),
                pair_acc_mean=("pair_acc", "mean"), pair_acc_std=("pair_acc", "std"),
                hap_acc_mean=("hap_acc", "mean"), hap_acc_std=("hap_acc", "std"),
                homo_pred_mean=("homo_pred", "mean"))
           .reset_index()
           .sort_values(["n_founders", "n_breakpoints_per_chrom"]))
    float_cols = {c for c in agg.columns if c.endswith(("_mean", "_std"))}
    detail_md = _markdown_table(agg, float_cols)

    overall = (detail.groupby("n_founders")
               .agg(n_runs=("seed_idx", "count"),
                    pair_acc_mean=("pair_acc", "mean"), pair_acc_std=("pair_acc", "std"),
                    hap_acc_mean=("hap_acc", "mean"))
               .reset_index()
               .sort_values("n_founders"))
    overall_float_cols = {c for c in overall.columns if c.endswith(("_mean", "_std"))}
    overall_md = _markdown_table(overall, overall_float_cols)

    model_desc = ("the **affinity-conditioned** diploid GRITS-CRF "
                   "(`checkpoints/diploid-affinity-sim512-h3`)" if affinity else
                   "the plain (non-affinity) diploid GRITS-CRF")
    lines = [
        f"# N-way recombination Tripsacum baseline ({'affinity' if affinity else 'plain'} "
        f"diploid CRF) -- {N_SEEDS} seeds x {{4,6}} founders, averaged\n",
        "Generalizes the 2-way recombination sweep (`tripsacum_recombination.md`) from "
        "exactly 2 founders (strict A/B alternation) to a pool of N=4 or N=6 founders -- "
        "every distinct individual used anywhere in the pair-based tests, recombined "
        "N-way instead of 2-way. H1 and H2 each independently draw a new founder from the "
        "pool at each breakpoint (constrained only to differ from the immediately "
        "preceding segment), so the true pair at any site can be any 2 of the N founders "
        "(or the same one twice -- homozygous). Founder feature columns are masked to the "
        f"locally-true pair before scoring with {model_desc}. {N_SEEDS} independent random "
        "breakpoint/founder-assignment draws are averaged per (founder-set, level).\n\n"
        "## Overall (averaged across all levels and seeds, per founder-set size)\n\n"
        f"{overall_md}\n\n"
        "## Per (founder-set, level)\n\n"
        f"{detail_md}\n\n"
        f"Full per-seed detail: `{detail_tsv.name}`.\n",
    ]
    results_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {results_md}")
    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nPer (founder-set, level):")
    print(agg.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]

    if mode == "list":
        print(f"Founder sets: {FOUNDER_SETS}")
        print(f"Depth: {DEPTH:,}/hap")
        print(f"Levels (breakpoints/chrom): {LEVELS}")
        print(f"Seeds per (founder-set, level): {N_SEEDS}")
        for n, founders in FOUNDER_SETS.items():
            for f in founders:
                try:
                    td.assembly_path(f)
                except FileNotFoundError as e:
                    print(f"  N={n}: MISSING -- {e}")
                    break
            else:
                print(f"  N={n}: {'+'.join(founders)}  OK (assemblies present)")
        print(f"\nTotal runs: {len(FOUNDER_SETS)} founder-sets x {len(LEVELS)} levels x "
              f"{N_SEEDS} seeds = {len(FOUNDER_SETS) * len(LEVELS) * N_SEEDS}")
        return

    if mode == "report-nway":
        write_report()
        return
    if mode == "report-nway-affinity":
        write_report(AFFINITY_DETAIL_TSV, AFFINITY_RESULTS_MD, affinity=True)
        return

    if mode in ("one-nway", "one-nway-affinity"):
        if len(sys.argv) < 5:
            raise SystemExit(f"usage: tripsacum_recombination_nway.py {mode} <n_founders> <n_breakpoints> <seed_idx>")
        n_founders, n, seed_idx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = td.AFFINITY_CKPT if mode == "one-nway-affinity" else td.DIPLOID_CKPT
        tsv = AFFINITY_DETAIL_TSV if mode == "one-nway-affinity" else DETAIL_TSV
        write_header_if_needed(tsv)
        run_one(n_founders, n, seed_idx, device, force=True, ckpt_path=ckpt, detail_tsv=tsv)
    elif mode in ("all-nway", "all-nway-affinity"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = td.AFFINITY_CKPT if mode == "all-nway-affinity" else td.DIPLOID_CKPT
        tsv = AFFINITY_DETAIL_TSV if mode == "all-nway-affinity" else DETAIL_TSV
        write_header_if_needed(tsv)
        for n_founders in FOUNDER_SETS:
            for n in LEVELS:
                for seed_idx in range(N_SEEDS):
                    run_one(n_founders, n, seed_idx, device, ckpt_path=ckpt, detail_tsv=tsv)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {AFFINITY_DETAIL_TSV if 'affinity' in mode else DETAIL_TSV}")


if __name__ == "__main__":
    main()
