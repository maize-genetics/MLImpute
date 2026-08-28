# Whole-chromosome CRF decode: verification on chr-mosaic RIL + real inbreds

## What was built

`src/python/crf/infer_wholegenome_real.py` (new, on branch
`wholechrom-decode-real-data` off `develop`, worktree
`grits-wholechrom-worktree`): reuses E12's `infer_wholegenome.py`
`decode_chrom`/`_starts`/`_ownership`/`affinity_keep` **unmodified** —
sliding overlapped-window encode (win=512, stride=256, matched to the
`diploid-affinity-sim512-h3` checkpoint's T=512 training length, not
E12's 1024/512 defaults), hard center-crop stitch of `emis_p`/`c` into one
`[1,L,P]`/`[1,L]` track per chromosome, single `_dcrf_viterbi` decode over
the whole thing. Two additions on top:

1. **Sparse per-contig loading** (`load_real_contigs`): reads
   `raw.npy`/`raw.npy.bins.tsv` directly, groups by contig, applies the
   same fixed K25→K24 drop (P39/idx23) as `simval_eval_one.window_fixed_drop`
   — ragged per-contig arrays, no window-size truncation (every real bin
   is used, not just window-aligned ones).
2. **`homo_scale` plumbed through** (`_encode_real`, monkey-patched onto
   `infer_wholegenome._encode` at runtime, no edit to that shared file):
   the original always calls `model(Xb, None, eb)`, silently applying the
   checkpoint's *full* fixed `homo_penalty` — exactly the mismatch that
   caused 39% error on the chromosome-mosaic RIL earlier today. Now set
   per-row via `--homo-scale`.

BED output writer (`write_wholechrom_bed`) mirrors
`heldout_assembly_eval.write_imputed_bed`'s contract but consumes a flat
per-contig decoded path instead of a `[n_windows,T]` reshape.

## Results — same per-base founder-error metric as today

| row | homo_scale | **per-window (today)** | **whole-chrom decode** |
|---|---|---|---|
| chr-mosaic RIL | 0.5 | 39.22% error, 341-889 intervals/chrom | **38.78% error, 313-836 intervals/chrom** |
| chr-mosaic RIL | 0.0 | 0.036% error, 1/chrom + 2 boundary blips (chr3 22.8kb, chr6 742kb) | **0.036% error, 1/chrom + same 2 blips (chr3 ~25kb, chr6 ~740kb)** |
| real Oh43 inbred | 0.0 | 0.0018% error | **0.0000% error** |
| real Il14H inbred | 0.0 | 0.0028% error | **0.0026% error** |

Decode was fast and stable: 2.9-10.6s per chromosome (Viterbi over
79k-158k real sites), no memory issues, no crashes, all 4 runs completed
cleanly end to end (align → decode → BED) on the first try.

## Interpretation

**Whole-chromosome context does not fix either of the two error sources
diagnosed today — but it also doesn't hurt anything.**

- **The `homo_scale=0.5` mismatch is essentially unchanged** (39.22% →
  38.78%). This makes sense mechanistically: a globally wrong scalar
  prior biases the emission score for *every* candidate state at *every*
  site roughly uniformly, so there's no "correct majority outvoting a
  local error" effect for chromosome-scale context to exploit — the whole
  chromosome is uniformly miscalibrated, not locally confused. The fix
  for this case is still simply using the correct prior, not more
  context.
- **The chr3/chr6 boundary blips (P39/Il14H-relatedness-driven local read
  clustering, from earlier today) persist essentially unchanged in both
  location and magnitude** under whole-chromosome decode. The hypothesis
  that surrounding correctly-called context would out-compete a
  locally-dominant misleading read cluster did not hold: the CRF's
  transition cost is paid per switch-event, not per "distance from a
  locally strong wrong answer," so a sufficiently locally-confident wrong
  emission can still win its own site even inside a much longer correct
  context — the recombination cost `c` is itself computed locally per
  site and isn't necessarily elevated just because the surrounding
  chromosome is stable.
- **The two real inbred rows improved slightly** (Oh43 to exactly 0.0000%,
  Il14H nearly unchanged at 0.0026%) — consistent with removing the small
  number of tile-boundary artifacts that non-overlapping per-window
  decode can produce at a window edge, matching E12-edge's own small,
  positive finding on synthetic data (+0.0008 to +0.0094 pair_acc at
  boundary sites). This is a genuine, if modest, benefit of the
  architecture change on data that was already near-perfect.

## Verdict

Whole-chromosome decode is now a working, verified capability for real
sparse corpus data (not wired into the production `simval_eval_one.py`
pipeline yet, per scope) — implemented cleanly by reusing E12's existing
stitching machinery almost entirely unmodified. It's a real, if modest,
win on already-good data (boundary-artifact cleanup) but is **not** a fix
for either of today's two diagnosed error sources: the `homo_scale`
mismatch needs the correct prior, not more context, and the
sparse-coverage/IBD-relatedness local read clusters are locally dominant
enough that even whole-chromosome-scale surrounding evidence doesn't
override them. Confirms that `homo_scale` calibration remains the
dominant, first-order lever for real accuracy — architecture changes to
decode scope are a secondary, smaller effect.
