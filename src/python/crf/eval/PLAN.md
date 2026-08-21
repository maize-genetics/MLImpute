# Windowing-step quality filters: fan-out drop + binarization

> Scope note: this is a small, self-contained plan for one feature, not the
> main `docs/PLAN.md` experiment series. See `RESULTS.md` in this directory
> for what was actually measured, and `HANDOFF.md` for how to run it and
> what's still open.

## Context

A founder-error classification on a real `IDX-RIL2` Oh43xIl14H 0.1x
coverage row (a true single-crossover-path RIL individual, from the maize
simulated-validation read corpus) found the dominant decode error mode is
large (5kb–1.8Mb), isolated, confidently-homozygous "phantom parent-swap"
intervals far from any true recombination breakpoint (~92% of all wrong
bp), not boundary smearing at real crossovers. Root-causing this surfaced
two real, fixable data-quality issues in the windowing step
(`ropebwt_npy_to_matrix.py`, which converts refmap's per-bin per-founder
read-count matrix into the model's input):

1. **No filter on read informativeness.** `raw.npy` is written one row per
   unique `(contig, bin, gameteSet)` triple (confirmed in `ps4g.c`, by
   design, so co-occurrence info isn't lost). A read's `gameteSet` fan-out
   (how many of the 25 founders it matches) is fully preserved per row —
   `(row != 0).sum()` reproduces the exact gameteSet size. Reads matching
   a large fraction of the panel are close to uninformative (they don't
   discriminate between founders) and currently flow straight into the
   feature matrix unfiltered. `--max-occ` (the existing refmap flag) does
   NOT do this — it caps total pangenome occurrence count, not founder
   diversity, and at `--max-occ=-1` (this pipeline's setting, resolves to
   25) a read hitting all 25 founders once each is kept as a fully
   "EXACT" 25-nonzero-column row.

2. **Train/eval feature-scale mismatch.** The model
   (`diploid-affinity-sim512-h3`) was trained exclusively on **binary
   0/1** features (every training `.npy`'s unique feature values are
   `[0, 1]`; `simulate_alleles.py`'s docstring and match-computation
   confirm this by construction). The real-data windowing script instead
   passed raw read counts straight through (`np.clip(arr[:,:K],0,127)`,
   observed real values up to 15-70). The model's `cell(log1p(X))`
   encoder path and `depth = log1p(X.sum(-1))` recombination-cost feature
   both treat this as a continuous magnitude, not a match indicator — an
   extrapolation into input space the model never saw in training. Two
   more consumers explicitly assume bounded 0/1 input and silently break
   on counts: `_founder_affinity` (docstring: "feats_block (binary
   match)", used by both affinity checkpoints) and `_het_scale` (a
   Jaccard, only valid for 0/1). Binarizing at the windowing step fixes
   all of these at once, at the one place features are materialized.

Goal: add both as tunable knobs at the windowing step, defaulting per
user intent (binarize ON by default since it's a correctness fix; fan-out
filter OFF by default / opt-in since it measurably reshapes window
boundaries and should be validated before becoming a silent default
everywhere), then validate on the real `IDX-RIL2` Oh43xIl14H 0.1x row
(reusing its already-built `raw.npy`/`raw.ps4g` — no refmap re-run
needed) and compare founder error against the pre-change baseline using a
bp-weighted true-mosaic comparison.

## Design

### 1. `src/python/crf/ropebwt_npy_to_matrix.py` (primary edit)

This is the one place `raw.npy` counts get materialized into the model's
feature dtype/shape — both features belong here per the file's own stated
scope ("the standard grits training-matrix contract").

- New args: `--max-hit-frac FLOAT` (default `None` = off) and
  `--retain-counts` (`store_true`, default `False` → binarize by
  default).
- After the existing dtype-cast/label-remap, insert, in order:
  1. **Fan-out filter** (only if `--max-hit-frac` given): `fanout =
     (feats != 0).sum(axis=1)`; `max_founders = max(1, int(max_hit_frac *
     K))` using the SOURCE K (25, computed before any
     `--target-num-parents` trim, so "half the assemblies" means half of
     the real 25-founder panel regardless of what a caller later drops
     down to); `keep = fanout <= max_founders`; subset `feats`, `gA`,
     `gB`, **and** `bins_df` together — row-dropping shifts every
     downstream window boundary (windowing is row-count-based, not
     position-based), so `bins_df` must stay in lockstep or genomic
     coordinates silently desync.
  2. **Binarize** (unless `--retain-counts`): `feats = (feats !=
     0).astype(np.int8)`.
- Write a filtered bins sidecar whenever `--max-hit-frac` is set:
  `Path(str(args.out) + ".bins.tsv")`, same 3-column schema as the input
  `--bins` file, containing the post-filter (pre-window-truncation) row
  set — this is what lets `heldout_assembly_eval.load_contig_layout`/
  `write_imputed_bed` (which independently re-derives window/contig
  layout from a bins.tsv file) reproduce the exact same window
  boundaries this script produced.

### 2. `eval/heldout_assembly_eval.py::window()`

- New params `max_hit_frac=None, retain_counts=False`, threaded into the
  subprocess `cmd` list.
- **Filename-safety fix:** extend the existing `suffix` computation:
  append `_bin` when `not retain_counts` (the new default — a fresh
  filename forces correct recompute instead of the existing cache check
  silently returning a stale count-valued array under the old filename),
  and `_hitfrac{max_hit_frac}` when set. `--retain-counts` alone
  reproduces the exact old filename/cache byte-for-byte.
- No return-signature change — every other caller of this shared
  function keeps working unmodified.

### 3. `eval/simval_eval_one.py`

- `window_fixed_drop(raw_npy, outdir, drop_idx, max_hit_frac=None,
  retain_counts=False)`: threads the two new params through; extends its
  own output filename with the same suffix convention; returns
  `(out_path, bins_path)` where `bins_path` is the filtered sidecar when
  `max_hit_frac` is set, else the original `raw.npy.bins.tsv`.
- `do_align(args)`: threads `args.max_hit_frac`/`args.retain_counts`
  through both windowing branches; uses the resolved `bins_path` (not a
  hardcoded `outdir / "raw.npy.bins.tsv"`) when writing the imputed BED.
- New CLI args: `--max-hit-frac` (float, default `None`), `--retain-
  counts` (`store_true`).

### 4. Validation run

- Reuse the already-built `raw.npy`/`raw.ps4g` for `IDX-RIL2
  __Oh43xIl14H__0.1x` (`run_refmap` is resumable — skips refmap entirely
  when `raw.npy`+`raw.tsv` exist).
- Sweep `--max-hit-frac` (off, 0.5, 0.3, 0.2, 0.1) with binarize default
  on, recomputing bp-weighted founder error against the row's true
  (seeded, reproducible) recombination mosaic each time — see
  `RESULTS.md` for numbers.

## Verification

- `ropebwt_npy_to_matrix.py` printout confirms rows dropped (count +
  percentage) when `--max-hit-frac` is set, and the output array's
  unique feature values are exactly `{0, 1}` when binarized.
- New `.bins.tsv` sidecar exists next to the filtered windowed npy, same
  row count as the filtered (pre-window-truncation) feature array.
- Re-running with `--retain-counts` and no `--max-hit-frac` (old
  behavior) hits the existing cache filename and returns instantly,
  confirming the default change doesn't silently invalidate or corrupt
  anything already on disk. (Verified: resolves to the byte-identical old
  `windowed_k25.npy` path and returns without recompute.)
- Founder-error recompute against the true RIL2 mosaic (genome-wide +
  per-chromosome), swept across thresholds — see `RESULTS.md`.
