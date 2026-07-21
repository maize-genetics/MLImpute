# Sweeping `--kmer` placement parameters (length, step, min-agree, cluster)

Follow-up to `kmer75_vs_wholeread.md`. That single trial (k=75, step=15, agree=2,
cluster=2000 — all documented defaults) gave Viterbi accuracy 0.2352, far below whole-read
mode's 0.9915. Question: is that a bad parameter choice, or a structural ceiling for
k-mer-agreement placement on this data? Swept `--kmer` (length), `--kmer-step`,
`--min-agree`, staged one-factor-at-a-time (not full factorial — a 5×5×5 grid would be ~125
refmap runs, impractical). `--kmer-cluster` (Stage D) was skipped — Stages A-C already gave
a clear, consistent, plateauing picture that a cluster-size sweep was very unlikely to
overturn.

Driver: `grits_workdir/scripts/kmer_sweep.py` (loads the checkpoint once, reuses it across
all combos; caches/skips any combo whose output already exists). Same benchmark set,
checkpoint, and windowing pipeline used throughout this investigation. Full data:
`grits_workdir/results/kmer_sweep_results.tsv`.

## Stage A — kmer length (step=15, agree=2, cluster=2000 fixed)

| k | viterbi | Oh43-covered | mean founders/site | Oh43 coverage (K25) |
|---:|---:|---:|---:|---:|
| 25 | 0.2150 | 33.3% @ 0.525 | 2.45 | 32.2% |
| 50 | 0.2359 | 35.8% @ 0.541 | 2.45 | 33.5% |
| 75 | 0.2380 | 36.2% @ 0.545 | 2.40 | 33.8% |
| 100 | 0.2615 | 36.9% @ 0.592 | 2.33 | 34.5% |
| **125** | **0.2777** | 37.1% @ 0.627 | 2.24 | 34.8% |

Clear, monotonic trend: **longer k-mers do better**, even though founder-support density
stays essentially flat (~2.2-2.45 founders/site) or even drifts slightly down. Longer
k-mers are more specific/unique matches, so each vote is higher-confidence — the gain here
is precision-of-placement, not density-of-support. (Small note: the k=75 point here,
0.2380, reproduces the earlier manual run's 0.2352 up to ~0.3 percentage points — this
script's per-item decode loop vs. `crf/eval.py`'s batched `DataLoader` give mathematically
equivalent but not bit-identical aggregation; negligible for ranking purposes, everything in
this sweep uses this script consistently.)

## Stage B — kmer step (k=125, agree=2, cluster=2000 fixed)

| step | viterbi | mean founders/site |
|---:|---:|---:|
| 5 | 0.2745 | 2.23 |
| 10 | 0.2747 | 2.24 |
| 15 | 0.2777 | 2.24 |
| **25** | **0.2832** | 2.24 |
| 40 | 0.2749 | 2.26 |

**Step barely matters** — everything sits in a narrow 0.2745-0.2832 band with no real
trend, and founder density is essentially step-invariant. This argues against the original
"denser tiling recovers density" hypothesis: at k=125 (already close to the 151bp read
length), the tile-start range is only ~26bp regardless of step, so successive tiles heavily
overlap no matter how finely they're spaced — step just isn't the lever it looked like from
the min-agree explanation alone. step=25 is the nominal best, by a small, possibly-noisy
margin.

## Stage C — min-agree (k=125, step=25, cluster=2000 fixed)

| agree | viterbi | n_placed |
|---:|---:|---:|
| 1 | 0.2751 | 596,398 |
| **2** | **0.2832** | 546,048 |
| 3 | 0.2741 | 499,347 |
| 4 | **degenerate — 0 placed** | 0 |
| 5 | **degenerate — 0 placed** | 0 |

Feasible range (1-3) is flat, same story as step. **agree=4 and 5 are structurally
impossible at k=125/step=25**: a 151bp read tiled at k=125/step=25 produces exactly 3
tiles (offsets 0, 25, plus a forced 3'-end tile) — `support` (distinct agreeing tiles) can
never exceed 3, so `min_agree >= 4` guarantees every single read stays UNPLACED. Confirmed
directly: `k125_s25_a4_c2000`'s raw TSV is 1,000,000/1,000,000 UNPLACED. This is a hard
ceiling relating `min-agree` to `(read_length - k)/step + 2`, not tunable independently of
k/step — worth knowing if trying larger `min-agree` values elsewhere.

## Overall winner: k=125, step=25, agree=2, cluster=2000 — viterbi=0.2832

Still **far below whole-read mode's 0.9915** (about a third of it). Every combination
tested caps Oh43 coverage around 32-35% (vs. whole-read's 95.6%) — this ceiling, not any of
the four swept parameters, is the actual bottleneck, and nothing in this grid broke through
it.

## Independent-truth validation of the winner

Scored the winning combo's raw refmap TSV through the same `validation/run_all.sh` pipeline
used for the kmer75-vs-whole-read comparison (same cached truth, no recompute):

| | kmer75 (original) | **kmer125/s25/a2 (winner)** | whole-read (fixed) |
|---|---:|---:|---:|
| Truth-having reads called | 87.9% | 80.0% | 96.3% |
| Chromosome-correct (of called) | 97.73% | **98.49%** | 98.13% |
| Correct @ 1000bp (best bucket) | 14.9% | 15.1% | 97.2% (EXACT) |

A genuine, coherent trade-off: the winning combo has **better placement precision** than
the original kmer75 trial (fewer wrong-chromosome calls — even edging out whole-read mode's
PLACED+EXACT blend) but **worse recall** (fewer reads get a call at all — longer, stricter
k-mers are pickier). Consistent with the model-accuracy story: modestly better than kmer75,
nowhere near whole-read mode, because whole-read mode's `EXACT` fast path (30% of reads,
93-98% precise at the exact base) has no k-mer-mode equivalent at any of these settings.

## Bottom line

No combination of length/step/min-agree tested closes the gap to whole-read placement on
real Oh43 data. Longer k-mers help modestly and consistently (worth using k≈100-125 over the
originally-tried 75 if k-mer mode is used at all); step and min-agree matter little in the
feasible range once k is large. The real limiting factor is structural: k-mer-agreement
placement, as implemented, has no equivalent to whole-read mode's `EXACT` reference-match
fast path, and per-read founder-support density stays capped well below what whole-read mode
achieves regardless of these four parameters. Not tested: `--kmer-cluster` (Stage D, skipped
as low-yield given the clear plateau already found), or k-mer mode's precision advantage
under higher simulated sequencing-error rates (the condition it was actually designed for,
per the synthetic `e4` benchmark) — real Oh43 reads may simply be clean enough that this
mode's error tolerance isn't needed.
