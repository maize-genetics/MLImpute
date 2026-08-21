# Handoff — windowing fan-out filter + binarize-by-default

Branch: `windowing-quality-filters` (off `origin/tripsacum-tests` — **not**
`develop`; `ropebwt_npy_to_matrix.py` only exists on `tripsacum-tests`,
not yet merged, so branch from there or you won't have the file to edit).
2 commits: the core `ropebwt_npy_to_matrix.py` feature, then this
`src/python/crf/eval/` directory (validation driver + its pipeline
dependencies, copied in from a separate non-git scratch tree — see
"Why eval/ has copies of shared scripts" below). Not yet opened as a PR.

## What's here

- `ropebwt_npy_to_matrix.py` (in `src/python/crf/`, one level up from this
  directory) — the actual feature: `--max-hit-frac` and `--retain-counts`
  flags.
- `run_ril2_windowfilter_test.py` — the validation driver used to produce
  every number in `RESULTS.md`. Reuses an already-built real corpus row's
  `raw.npy`/`raw.ps4g` (symlinked in, refmap is resumable so it's never
  re-run) and drives the align stage (window → diploid-affinity CRF
  decode → founder-path BED) via `simval_eval_one.do_align`.
- `heldout_assembly_eval.py`, `simval_eval_one.py`, `simval_paths.py`,
  `nam_baseline.py` — unmodified copies of the shared pipeline scripts
  `run_ril2_windowfilter_test.py` depends on (see below for why they're
  copies, not imports from elsewhere).
- `PLAN.md` — why this was built and the design.
- `RESULTS.md` — the actual sweep numbers and interpretation.

## How to re-run the sweep

```
cd src/python/crf/eval
LD_LIBRARY_PATH= PYTHONPATH=/path/to/this/checkout/src \
  /home/zrm22/mambaforge/envs/phg-ml/bin/python run_ril2_windowfilter_test.py \
    --max-hit-frac 0.2
```

This expects (hardcoded paths in the script, matching the original
grits_workdir scratch layout it was developed against):
- A pre-built real corpus row at `.../grits_workdir/scratch/simval_eval/
  IDX-RIL2__Oh43xIl14H__0.1x/` (raw.npy, raw.npy.bins.tsv,
  raw.npy.gametes.tsv, raw.ps4g, raw.tsv, raw.log) — from a prior
  `simval_eval_one.py --stage align` run on that row.
- The `diploid-affinity-sim512-h3` checkpoint and v2 panel VCF at the
  paths in `simval_paths.py`.

Output lands in a fresh sibling directory,
`IDX-RIL2__Oh43xIl14H__0.1x__hitfrac{X}-bin/`, with `bed/*.imputed.bed`
per chromosome — same contract as every other row this pipeline produces.

Founder error against the row's true mosaic (not included as a script
here — was run ad hoc during development) does roughly:
```python
from simval_oracle_bed import build_ril_mosaics   # elsewhere in grits_workdir/scripts
mosaic_h1, mosaic_h2, label_to_name = build_ril_mosaics("IDX-RIL2", "Oh43", "Il14H")
# then bp-weighted compare each decoded BED interval's (parent1,parent2)
# against the true founder at that position (bisect into mosaic_h1[chrom]'s
# sorted (start,end,label) segments), splitting decoded intervals at any
# true breakpoint they straddle.
```
`simval_oracle_bed.py` itself (which has `build_ril_mosaics` and the
sibling `write_bed_ril_exact`) was NOT copied into this directory — it's
a different concern (writing ground-truth BEDs for chain-error isolation)
that happens to share the mosaic-reconstruction helper. If this comparison
becomes a permanent fixture, port `build_ril_mosaics` (and a small
bp-weighted diff function) into a real script here instead of re-deriving
it ad hoc each time.

## Why `eval/` has copies of shared scripts, not imports

`heldout_assembly_eval.py`/`simval_eval_one.py`/`simval_paths.py`/
`nam_baseline.py` normally live in `grits_workdir/scripts/`, a **separate,
non-git scratch working directory** (`/local/workdir/.../grits_workdir`),
not this repo. That's where the full maize evaluation corpus, checkpoints,
and every other row's outputs live, and where most of this codebase's
day-to-day evaluation scripts have accumulated across many sessions —
only these five were brought into git, scoped to just this feature, per
explicit user request (not a wholesale "graduate everything" move). They
import each other via `sys.path.insert(0, str(Path(__file__).parent))`,
which still works unchanged now that they're all copied into the same
`eval/` directory — verified by importing `simval_eval_one`/
`run_ril2_windowfilter_test` directly from this location before
committing.

**Known consequence:** these are now **duplicated** — grits_workdir's
copies are the ones actually used for day-to-day corpus work, and won't
automatically pick up future edits made here (or vice versa). If more of
the shared pipeline evolves, decide explicitly whether `grits_workdir`'s
scripts should start importing from this repo location instead of staying
independent copies.

## Gotchas

- `ropebwt_npy_to_matrix.py`'s binarize-by-default change is
  **repo-wide**: every other caller of `heldout_assembly_eval.window()`
  (`nam_diploid.py`, `cassava_diploid.py`, `tripsacum_diploid.py`, all
  still only in `grits_workdir/scripts/`, not touched by this branch)
  will pick up binarized features + a new (`_bin`-suffixed) cache
  filename the next time they run, with zero code changes needed on
  their end. This is intentional (fixes a real train/eval mismatch — see
  `PLAN.md`) and non-destructive (old cached count-valued `.npy` files
  are simply no longer looked up, never deleted) — but it does mean any
  results from those other pipelines generated *after* this branch is
  merged are not directly comparable to anything generated before it
  without checking which feature convention was in effect.
- `--max-hit-frac` is **not yet the pipeline default** anywhere — every
  existing caller keeps its old (unfiltered) behavior unless explicitly
  passed the new flag. Binarize, by contrast, IS now the default the
  moment this branch merges.
- The K used for the fan-out fraction is always the SOURCE panel size
  (25), computed before any `--target-num-parents` trim — "half the
  assemblies" means half of the real 25-founder panel, not whatever a
  caller later reduces it to (e.g. K24 after the fixed P39 drop).

## Open next steps

1. **Confirm the `--max-hit-frac 0.2` optimum generalizes** — only tested
   on one individual (`IDX-RIL2` Oh43xIl14H, 0.1x). Sweep at least one or
   two other `IDX-RIL2` pairs (and maybe a coverage rung other than 0.1x)
   before treating 0.2 vs. 0.5 as more than a single-row observation. The
   `--max-hit-frac 0.1` reversal (evidence starvation at true breakpoints)
   is itself a coverage-density effect, so the safe/optimal threshold
   likely shifts with coverage — worth checking directly.
2. **Decide on a pipeline default for `--max-hit-frac`.** Per user
   judgment (2026-08-21): production will likely use `0.5` (safely away
   from the 0.1 cliff) rather than the empirically-best-on-this-row `0.2`.
   Not yet wired as a default anywhere.
3. **De-duplicate `eval/` vs. `grits_workdir/scripts/`** if this becomes
   a permanent fixture rather than a one-off validation (see above).
4. The RIL dataset itself is being independently rebuilt for more
   realism in a separate track of work (see Claude memory:
   `ril_dataset_rebuild_pending` / `simval_truth_labels_mechanism`) —
   once that lands, re-check whether this filter's optimal threshold
   holds on the new data.
