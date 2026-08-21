# Results — windowing fan-out filter + binarize-by-default

Validation row: real `IDX-RIL2` Oh43xIl14H, 0.1x coverage (from the maize
simulated-validation read corpus) — a true single-crossover-path RIL
individual (`kind=ril2`, homozygous per-locus, 28 genome-wide crossovers,
seeded/reproducible mosaic via `mosaic.derive_ril2_mosaic`). Checkpoint:
`diploid-affinity-sim512-h3` throughout, `--drop-idx 23` (P39). Founder
error = bp-weighted comparison of the decoded `parent1`/`parent2` BED
against the row's known true founder-path mosaic (see `HANDOFF.md` for the
exact comparison method — no gVCF/genotype comparator involved, so this is
fast and independent of the corpus's truth-gVCF pipeline).

## Fan-out filter sweep (binarize on throughout)

| `--max-hit-frac` | max founders/read | rows dropped | genome-wide error | wrong intervals |
|---|---:|---:|---:|---:|
| off (baseline) | 25 | 0% | 0.478% | 89 |
| 0.5 | 12 | 31.3% | 0.420% | 83 |
| 0.3 | 7 | 50.5% | 0.382% | 75 |
| **0.2** | **5** | **60.5%** | **0.263% (best)** | 64 |
| 0.1 | 2 | 82.5% | 0.365% (worse) | 57 |

**Not monotonic.** Error improves steadily as the threshold tightens from
off through 0.2, then reverses at 0.1.

## Error-mode breakdown (wrong bp, split boundary-adjacent vs. isolated)

Boundary-adjacent = within 20kb of a true recombination breakpoint;
isolated = everywhere else, further split by whether the wrong call is the
*other real parent* of this individual (Oh43↔Il14H) vs. some unrelated
third founder (a relatedness/PAV-type miscall, e.g. the P39/Il14H IBD
artifact found earlier in this investigation).

| `--max-hit-frac` | boundary-adjacent bp | isolated bp (total) | — other-true-parent swap | — unrelated founder |
|---|---:|---:|---:|---:|
| off | 596k | 9.3M | 9.3M (54 intervals) | small (5 intervals) |
| 0.5 | 880k | 8.0M | 8.0M (22) | tiny (1) |
| 0.3 | 724k | 7.4M | 7.4M (18) | 1 bp (1) |
| 0.2 | 999k | 4.5M | 4.5M (13) | **0** |
| 0.1 | **3.86M** | 3.8M | 3.8M (11) | 0 |

## Interpretation

- **The fan-out filter's clearest, cleanest win is eliminating relatedness/
  PAV-type miscalls** (reads that are genuinely ambiguous across many
  founders, matching a wrong-but-related founder like the earlier P39/
  Il14H case) — down from 5 intervals at baseline to 0 by threshold 0.2.
- **The dominant "isolated large-stretch wrong-parent" error mode DOES
  respond to filtering, more than first thought.** It shrinks steadily
  with tighter thresholds (9.3M→8.0M→7.4M→4.5M→3.8M bp from off→0.1), so
  a meaningful fraction of it is genuinely noisy-read-driven, not purely
  an artifact of low coverage.
- **The reversal at 0.1 is a distinct, second failure mode:** boundary-
  adjacent error at real recombination breakpoints explodes 4x
  (999k→3.86M bp) once 82.5% of rows are dropped (max 2/25 founders
  survive). Too few reads remain near true crossovers to localize them
  precisely — the same "evidence starvation" mechanism that made a
  10k-read-subsample control (a separate, earlier check on a different
  synthetic RIL row) perform *worse*, not better, at ultra-low coverage.
  Filtering too aggressively recreates that problem by a different route.
- **0.2 is the best-performing threshold on this row** (0.263% error, a
  ~45% relative reduction from baseline), sitting at the inflection point
  before the breakpoint-starvation cost outweighs the noise-reduction
  benefit. **Not yet confirmed to generalize** — only tested on one
  individual/coverage; the exact optimum is plausibly coverage- and
  panel-relatedness-dependent.
- **Practical recommendation (user judgment, 2026-08-21):** production use
  will likely default to `--max-hit-frac 0.5`, safely away from the 0.1
  cliff, rather than the empirically-best-but-untested-elsewhere 0.2. This
  result also reconfirms an old imputation-modeling intuition from HMM-era
  work: restricting to informative/known founder hits measurably removes
  model-confusing noise, independent of model architecture.

## Column (founder) impact of the fan-out filter, `--max-hit-frac 0.5`

Zero founder columns were fully eliminated by filtering — every founder
retains support. But the reduction is very uneven, tracking relatedness to
the individual's two true parents (Oh43, Il14H):

| founder group | nonzero-cell reduction |
|---|---:|
| Il14H (true parent) | 38.1% |
| Oh43 (true parent) | 39.3% |
| P39 (known relative of Il14H) | 47.1% |
| the other 22 unrelated founders | 58–70% each |

Total nonzero `(row, founder)` cells: 10,203,293 → 3,990,525 (60.9%
dropped) at `--max-hit-frac 0.5`. This is the expected signature of the
filter working correctly: founders genuinely present in the individual (or
closely related to a present founder) lose proportionally less support
than unrelated founders, whose signal mostly came from shared/low-
diagnostic-value reads.

## Binarize-by-default (separate from the fan-out sweep)

Not separately ablated on this row (binarize was on throughout the sweep
above) — the correctness case is architectural, not empirical: every
checkpoint this pipeline loads was trained exclusively on binary `[0,1]`
features (verified on the actual training `.npy` files), while real data
was feeding counts up to 15–70 straight through as a continuous magnitude
into `cell(log1p(X))`, the recombination-cost `depth` feature, and (more
severely, since these are unbounded genome-wide aggregates) `_founder_
affinity` and `_het_scale`, both of which assume bounded 0/1 input. See
`PLAN.md` for the full mechanism.
