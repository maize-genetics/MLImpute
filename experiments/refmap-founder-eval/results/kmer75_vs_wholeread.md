# `--kmer 75` vs whole-read placement, on real Oh43 reads

Follow-up to the carrier-cap fix (`oh43_absent_rootcause.md`). User asked to try
`ropebwt3-phg`'s k-mer-agreement placement mode (`--kmer INT`, tiles the read and places
where enough k-mers agree — designed to be error-tolerant) at k=75 (half the 151 bp read
length, the one point already benchmarked on synthetic data in
`experiments/ref-sensitivity/e4/RESULTS.md`), and compare to the whole-read results.

## Prerequisite: gameteSet export added to k-mer mode

`refmap_query_kmer()` in `search.c` never populated `r->gametes`/`r->n_gametes`, so k-mer-mode
placements were always skipped by the PS4G/npy accumulator — no founder-support data, so no
possible comparison to the CRF model-accuracy numbers. Implemented this (search.c only,
~20 lines): added a source-sequence field (`sx`) to `refmap_vote_t`, tracked the winning
vote cluster's index range (`best_i`/`best_j`) through the clustering loop (previously
computed but discarded), and — only for reads that actually reach the `min_agree` PLACED
threshold — populate `r->gametes` from the distinct founders backing that cluster's votes
(no manual dedup needed; `rb3_ps4g_acc_add` already sorts+dedupes on ingestion, same as
whole-read mode's EXACT path relies on). No changes needed to `ps4g.c` or
`crf/ropebwt_npy_to_matrix.py`.

**Verified working**: `kmer75.ps4g`'s per-gamete header totals show Oh43 clearly dominant
(197,649 vs ~40-60k for other founders) — the same shape as whole-read mode. Direct spot
check: read `...1796:2106` (chr2) — the whole-read-mode gameteSet at its locus is
`{Oh43, CML52, Oh7B, M37W, CML247, CML333, Ms71, Ki11, CML228, NC350, CML69}` (11 founders);
the k-mer-mode npy row at the same read's placed bin shows `{Oh43, Ms71}` — smaller (looser,
per-tile-union semantics as expected — see caveat below) but **Oh43 is correctly present**.

**Semantic caveat**: whole-read mode's gameteSet is "founders sharing the read's one longest
exact match as a single unit." K-mer mode's is "founders backing any vote in the winning
cluster" — a union across the read's ~6 independent 75bp tiles (step 15), each capped
separately by `--max-occ`. Looser and, as the numbers below show, much sparser per read.

## Setup

Same benchmark set throughout: `bench_ps4g_npy/reads/Oh43_1M.fastq` (1M-read Oh43 subsample),
same patched binary, same lift/ref-prefix/max-occ flags.
`--kmer 75 --kmer-step 15 --min-agree 2 --kmer-cluster 2000` (documented defaults, the one
point already benchmarked in e4).

## Result 1 — Speed

| | whole-read | kmer75 |
|---|---:|---:|
| Wall time | 36.2 s | 113.1 s |
| User CPU | 639.1 s | 2171.9 s |
| Peak RSS | 8.65 GB | 8.37 GB |

**~3.1x slower wall, ~3.4x more CPU** — expected: k-mer mode issues ~6 separate FM-index
queries per read (one per tile) instead of one whole-read query.

## Result 2 — Raw status mix (1M reads)

| status | whole-read | kmer75 |
|---|---:|---:|
| EXACT | 301,796 (30.2%) | — (no EXACT concept in this mode) |
| PLACED | 411,931 (41.2%) | 586,630 (58.7%) |
| MULTI | 265,438 (26.5%) | — |
| UNPLACED | 20,835 (2.1%) | 413,370 (41.3%) |

## Result 3 — Independent-truth placement accuracy

Scored both through the project's existing `validation/` pipeline
(`REFMAP_OUT=<path> bash scripts/run_all.sh`), reusing the same cached truth chain
(minimap2-to-Oh43 SAM + Oh43→B73 gVCF liftover) for a same-reads, same-truth comparison
(577,295 of the 1M reads have independent truth, identically for both runs).

| | whole-read | kmer75 |
|---|---:|---:|
| Truth-having reads that get a call | 556,160 / 577,295 (96.3%) | 507,610 / 577,295 (87.9%) |
| Chromosome-correct (of called) | 545,734 / 556,160 (98.13%) | 496,105 / 507,610 (97.73%) |
| Best bucket, correct @ 0 bp | 93.4% (EXACT, n=239,474) | 0.6% (PLACED, n=507,610) |
| Best bucket, correct @ 1 kb | 97.2% (EXACT) | 14.9% (PLACED) |
| Best bucket, correct @ 100 kb | 97.7% (EXACT) / 83.6% (PLACED) | 82.1% (PLACED) |

Whole-read mode has both **higher recall** (more truth-having reads actually get a call) and
**comparable-to-better chromosome accuracy**, and its EXACT bucket (reads matching B73
itself) gives a large fraction of near-perfect single-bp calls that k-mer mode's single
PLACED-only mechanism has no equivalent for. K-mer mode's own spotcheck (EXACT-only by
design) is empty — it never emits EXACT status.

## Result 4 — CRF model accuracy (same checkpoint, `checkpoints/haploid-sim/last.ckpt`)

Windowed via `crf/ropebwt_npy_to_matrix.py`, evaluated via `crf/eval.py`, same real Oh43
individual, same protocol as the carrier-cap validation.

| | whole-read (fixed) | kmer75 |
|---|---:|---:|
| Overall Viterbi accuracy | **0.9915** | **0.2352** |
| Oh43-covered sites | 95.60% @ 0.9971 | 36.18% @ 0.5448 |
| Oh43-absent sites | 4.40% @ 0.9783 | 63.82% @ 0.0641 |
| Mean founders per informative site | ~9 | ~2.3 |

This is the dominant story: k-mer mode's founder-support signal is far sparser (mean ~2.3
founders per site vs ~9), Oh43 itself is covered at barely a third of sites, and where it's
absent the model is essentially guessing (0.0641, barely above 1/25 ≈ 4% chance). The CRF
checkpoint was trained on simulated data whose density resembles whole-read mode's real
output much more closely than k-mer mode's.

## Bottom line

At this parameter setting (k=75, step 15, min-agree 2, cluster 2000), k-mer-agreement
placement is worse than whole-read placement on **every axis measured** on real Oh43 data:
slower (~3x), lower recall and no ultra-precise bucket in the independent-truth check, and
far worse CRF model accuracy due to much sparser founder-support signal. This doesn't rule
out k-mer mode being useful under different conditions it was actually designed for (higher
sequencing-error rates, where whole-read exact/SMEM matching degrades faster than k-mer
voting) — the synthetic `e4` benchmark showed k-mer mode's precision advantage growing with
injected error rate. Real Oh43 reads are apparently clean enough that whole-read mode's
EXACT fast path dominates. Not tested here: other k-mer/step/min-agree combinations, or
performance at higher simulated error rates on this real dataset.
