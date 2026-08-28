#!/usr/bin/env python
"""
Input-DATA-level (not the truth label, not predictions) autocorrelation
diagnostic. Redo of error_autocorrelation.py's ACF, but on a signal derived
directly from the raw.npy READ-SUPPORT feature matrix, instead of either
the model's error indicator or the position-derived truth label (both
tried and rejected as "not actually the input data" earlier this session).

CORRECTED SIGNAL (2026-08-25, user-flagged): earlier this used fanout>1
(read support touches ANY >1 of the 25 founders) as "ambiguous" -- wrong,
because in a 25-way pangenome index, matching multiple founders at once is
the EXPECTED, routine outcome of shared ancestry/conserved sequence, not
evidence of anything problematic (fired on ~91-93% of the genome, which is
not a meaningful signal). The decision-relevant question for THIS
individual's diploid genotype call is narrower: does the row's read
support fail to distinguish specifically between Oh43 and Il14H -- this
individual's own two true parents -- not "any founder in the panel". This
is exactly the definition the Su1-locus and chr5:189-202Mb spot checks
already used successfully (Oh43+Il14H co-occurrence, ~38-40% genome-wide
baseline, far more discriminating than the old ~91-93% fanout>1 rate).

Signal: ambiguous[row] = (feats[row, Oh43_idx] != 0) AND (feats[row,
Il14H_idx] != 0). Autocorrelation of `ambiguous` vs row-index lag, per
chromosome, per coverage -- tests whether rows genuinely confusable between
this individual's own two parents cluster in row order, purely from the
read data, independent of any model output or ground-truth mosaic. Also
reports, for context only (uses truth just to label rows, not as the
measured signal), what fraction of ambiguous rows still include the true
founder among their support.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_paths as P  # noqa: E402
from simval_truth_labels import bin_truth_labels  # noqa: E402

INDIVIDUAL = "Oh43xIl14H"
PARENT_A, PARENT_B = "Oh43", "Il14H"
SOURCE_K = 25


def acf(x, max_lag):
    x = x.astype(np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def run_length_stats(x):
    x = np.asarray(x, dtype=bool)
    edges = np.diff(np.concatenate(([0], x.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverages", nargs="*", default=["0.01", "0.1", "0.5", "1.0", "2.0"])
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--tag", default="unfiltered-bin",
                     help="row-order source: 'unfiltered-bin' (real position order) or a "
                          "posrand{bp}bp tag (that window size's jittered row order)")
    ap.add_argument("--json-out", default=None,
                     help="optional: write per-coverage ACF array + genome-wide run-length "
                          "array to this JSON file, for downstream plotting")
    cli = ap.parse_args()

    export = {}
    for coverage in cli.coverages:
        outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__{cli.tag}"
        bins_path = outdir / "raw.npy.bins.tsv"
        gametes_path = outdir / "raw.npy.gametes.tsv"
        base_dir = P.SCRATCH_ROOT / f"IDX-RIL2__{INDIVIDUAL}__{coverage}x__unfiltered-bin"
        if not gametes_path.exists():
            gametes_path = base_dir / "raw.npy.gametes.tsv"
        # The jittered outdir's own npy is the raw_posrand{bp}bp.npy sidecar
        # (see run_ril2_posrand_sweep.py); fall back to it if raw.npy itself
        # isn't there (jitter outdirs never write a plain raw.npy).
        npy_path = outdir / "raw.npy"
        if not npy_path.exists():
            candidates = list(outdir.glob("raw_posrand*.npy"))
            npy_path = candidates[0] if candidates else npy_path
        if not bins_path.exists() or not npy_path.exists():
            print(f"SKIP {coverage}x: missing {bins_path} or {npy_path}")
            continue

        gamete_names = hae.load_gamete_names(gametes_path)
        truth_labels, _ = bin_truth_labels(
            bins_path, "ril2", gamete_names, dataset_id="IDX-RIL2",
            individual=INDIVIDUAL, parent_a=PARENT_A, parent_b=PARENT_B)
        truth_idx = truth_labels[:, 0]  # h1 == h2 for ril2

        arr = np.load(npy_path, mmap_mode="r")
        feats = arr[:, :SOURCE_K]
        oh43_idx = gamete_names.index(PARENT_A)
        il14h_idx = gamete_names.index(PARENT_B)
        ambiguous = ((np.asarray(feats[:, oh43_idx]) != 0)
                     & (np.asarray(feats[:, il14h_idx]) != 0)).astype(np.int8)

        # context only: of the ambiguous rows, how many still include the
        # true founder among their support (vs. entirely missing it)?
        row_idx_all = np.arange(len(truth_idx))
        has_true_support = np.zeros(len(truth_idx), dtype=bool)
        valid = truth_idx >= 0
        has_true_support[valid] = np.asarray(feats)[row_idx_all[valid], truth_idx[valid]] != 0
        amb_with_true = int((ambiguous.astype(bool) & has_true_support).sum())
        n_amb = int(ambiguous.sum())

        bins_df = pd.read_csv(bins_path, sep="\t")
        print(f"\n=== {coverage}x / {cli.tag}  (n_rows={len(bins_df):,}, "
              f"ambiguous rows={n_amb:,} [{n_amb/len(bins_df):.2%}], "
              f"of those still incl. true founder={amb_with_true:,} "
              f"[{amb_with_true/n_amb:.2%}]) ===" if n_amb else
              f"\n=== {coverage}x / {cli.tag}  (n_rows={len(bins_df):,}, ambiguous rows=0) ===")
        print("contig".ljust(8) + "".join(f"lag{k:<6}" for k in range(1, min(6, cli.max_lag + 1)))
              + "  mean_run  n_runs")

        acf_by_contig = {}
        all_run_lengths = []
        for contig, idx in bins_df.groupby("contig", sort=False).indices.items():
            idx = np.sort(idx)
            amb_c = ambiguous[idx]
            if len(amb_c) < cli.max_lag + 1:
                continue
            a = acf(amb_c, cli.max_lag)
            acf_by_contig[contig] = a
            lengths = run_length_stats(amb_c)
            all_run_lengths.append(lengths)
            mean_run = lengths.mean() if len(lengths) else 0.0
            print(contig.ljust(8) + "".join(f"{v:<9.4f}" for v in a[:5])
                  + f"  {mean_run:<8.1f}  {len(lengths)}")

        pooled = np.mean(list(acf_by_contig.values()), axis=0)
        all_run_lengths = np.concatenate(all_run_lengths) if all_run_lengths else np.array([])
        print("MEAN".ljust(8) + "".join(f"{v:<9.4f}" for v in pooled[:5]))
        if len(all_run_lengths):
            pct = np.percentile(all_run_lengths, [10, 25, 50, 75, 90, 99])
            print(f"ambiguous-run length (rows), genome-wide: n_runs={len(all_run_lengths):,} "
                  f"mean={all_run_lengths.mean():.2f} p10={pct[0]:.0f} p25={pct[1]:.0f} "
                  f"median={pct[2]:.0f} p75={pct[3]:.0f} p90={pct[4]:.0f} p99={pct[5]:.0f} "
                  f"max={all_run_lengths.max():.0f}")

        if cli.json_out:
            export[coverage] = {
                "n_rows": int(len(bins_df)),
                "n_ambiguous": n_amb,
                "acf_mean": pooled.tolist(),
                "run_lengths": all_run_lengths.tolist(),
            }

    if cli.json_out:
        Path(cli.json_out).write_text(json.dumps({"tag": cli.tag, "coverages": export}, indent=1))
        print(f"\nwrote {cli.json_out}")


if __name__ == "__main__":
    main()
