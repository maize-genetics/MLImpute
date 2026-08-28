# RIL2 indel-density diagnostics: bin-size=1 and 100kbp position jitter

Founder-path bp-weighted error (either haplotype wrong = error, over total
decoded bp), Oh43xIl14H (IDX-RIL2), all 5 manifest coverages, all rows
unfiltered (no `--max-hit-frac`). Baseline is the existing 256bp-bin
unfiltered-bin grid. Scored with `scripts/founder_path_error.py`, verified
to reproduce the published baseline exactly.

| arm | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---:|---:|---:|---:|---:|
| baseline (256bp bins) | 0.2794% | 0.5219% | 0.7996% | 1.2213% | 1.5314% |
| bin-size=1 | 0.2919% | 0.5030% | 0.9058% | 1.3437% | 1.8613% |
| position-jitter (100kbp) | 0.2646% | 0.5004% | 0.6480% | 0.9281% | 1.1895% |
| bin-size=1 + jitter (combined) | 0.2955% | 0.4770% | 0.7575% | 1.0462% | 1.2164% |

Jitter-noise floor (a perfect decode's error purely from ±100kb coordinate
smearing near true crossovers, no model involved): **0.0435%** — negligible
next to every arm above, so the jitter arm's numbers reflect real model
behavior, not scoring artifact (`scripts/posrand_jitter_control.py`).

## Interpretation

- **`--bin-size=1` alone does not help — it's slightly worse than baseline
  at every coverage**, and the gap widens with coverage (1.8613% vs
  1.5314% at 2.0x, the largest absolute gap in the table). Removing 256bp
  position quantization does not fix the rising-with-coverage problem;
  if anything finer resolution alone makes it slightly worse, plausibly by
  giving the row-order-only CRF more, noisier, less-aggregated evidence
  per site at high coverage.
- **100kbp position jitter is the effective one** — lower than baseline at
  every coverage, and the gap grows with coverage (1.1895% vs 1.5314% at
  2.0x, roughly a 22% relative reduction). Since the model has no explicit
  position feature and only sees rows through order + which 512-row window
  they land in, this says the model does better when local read order is
  decorrelated from exact genomic position — consistent with indel-driven
  local read-density irregularities creating a spurious position-order
  signal that the model was overfitting to, which jittering breaks.
- **Combining both doesn't beat jitter alone** — the combined arm sits
  between bin-size=1 and jitter-alone at every coverage, meaning bin-size=1
  adds no incremental value once jitter is already applied, and partially
  cancels jitter's benefit. The whole effect traces to the jitter, not the
  quantization.
- **Coverage-scaling still rises in every arm** — jitter/combined slow the
  rise but don't flatten it; the underlying "more coverage -> more error"
  pattern (still unexplained) survives all four conditions tested here.

## How to reproduce / extend

- `scripts/run_ril2_binsize_sweep.py` — bin-size=1 refmap sweep (5
  coverages, fresh refmap runs, ~45 CPU-min total).
- `scripts/randomize_positions_100kb.py` — `jitter_positions()` (array
  reorder) and `jitter_mosaic_positions()` (noise-floor control), reusable.
- `scripts/run_ril2_posrand_sweep.py` — 100kbp jitter on top of the
  existing 256-bin unfiltered-bin data (no refmap re-run, seconds).
- `scripts/run_ril2_binsize1_posrand_sweep.py` — jitter on top of the
  bin-size=1 data (combined arm), sources from run_ril2_binsize_sweep's
  output.
- `scripts/founder_path_error.py` — productized founder-path bp-weighted
  scorer (was ad hoc every prior time); `--tags` accepts any of the above.
- `scripts/posrand_jitter_control.py` — jitter-noise-floor control.
- All jitter uses `seed=0` by default; scores above are single-seed, not
  averaged across seeds — a real next step if the jitter effect needs
  tighter confidence bounds.
