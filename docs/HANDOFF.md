# Session handoff — crf-relatedness (2026-06-24)

Status snapshot for resuming work. Full per-experiment detail is in
[`docs/RESULTS.md`](RESULTS.md); experiment spec in [`docs/PLAN.md`](PLAN.md).

**Branch:** `crf-relatedness` — clean, pushed to `origin` at the commit below.
All data/checkpoints/logs under `/workdir/esb33/`. Run via
`pixi run --environment gpu python …` (2× H200 visible; set
`CUDA_VISIBLE_DEVICES` to parallelize across GPU 0/1).

## Where things stand

### 1. Haploid vs HMM on REAL maize — DONE (E1-maize)
Data: `/workdir/zrm22/HackathonJun2026/NonCollapsedDatasets/singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy`
(N=1.35M, T=512, 26 cols = 24 founder read-counts + H1 + H2; inbred H1==H2).
Trained on a 200k subset (`--limit-n 250000` → 200k/25k/25k), 5 epochs, single seed.

Held-out **test founder accuracy** (`eval_test.py`, `eval_hmm.py`):
| arm | params | test acc | bp F1(±2) |
|---|---|---|---|
| CRF full (per-site c) | 5.07M | **0.682** | 0.351 |
| CRF small d128/L4 | 0.88M | 0.669 | 0.337 |
| CRF per-window c | 5.07M | 0.665 | 0.288 |
| CRF large d384/L12 | 22M | 0.662 | — |
| **HMM Li–Stephens (best)** | ~0 | **0.614** | 0.252 |
| CRF no-transition | 5.07M | 0.284 | — |

Key findings: CRF beats HMM by +6.7 (even the 0.88M model by +5.5). The win is
**entirely in low-recombination windows** (stratified: [0,2] bp +12.9; [6+] bp
HMM edges by 1.9 on likely-noisy dense windows). The HMM over-segments
(hallucinates ~4× too many breakpoints). `c` does NOT collapse on real data
(within-window sd 0.49), so per-site c > per-window c.

### 2. Diploid pair-state CRF — DONE, first result (E4-probe)
`train_diploid.py`: joint decode over P=325 founder pairs on the shared encoder.
New diploid sim mode: each site = one read from a random gamete
(`--gamete-balance`), so two paths interleave and must be phased.

Best model (`--homo-penalty 3`, ckpt
`/workdir/esb33/checkpoints/diploid-h3/d-epoch=04-val/loss=88.089.ckpt`):
**test pair_acc 0.614, hap_acc 0.743** on `sim_diploid_512.npy` (100k windows,
F=0, ≤4 crossovers/gamete, coalescent).

**Two bugs fixed (don't reintroduce):**
- No hard two-switch ban — independent chromosomes ⇒ transition cost −c·nsw,
  nsw∈{0,1,2}, a 2-switch costs exp(−c)².
- **Homozygous collapse:** single-read pair emission emis_f[i]+emis_f[j] is
  maximized by (a,a) of the observed founder ⇒ decode goes 100% homozygous
  without a het prior. `--homo-penalty` (subtract from homozygous pair-states,
  like `diploid_hmm`) fixes it: 0.04→0.61 pair_acc. Use penalty 3.

## Key files (all on this branch)
- `src/python/crf/train_haploid.py` — haploid CRF + encoder glue; `make_splits`
  (deterministic head-slice; `--limit-n` caps N), `--window-c`, `--no-transition`,
  `--run-name` (per-arm ckpt/log dirs).
- `src/python/crf/train_diploid.py` — diploid pair-state CRF (`--homo-penalty`).
- `src/python/crf/simulate_alleles.py` — sim; diploid via `--inbreeding 0`
  `--gamete-balance`; coalescent relatedness; `--recomb-span` hidden-rate (E2).
- `src/python/crf/eval_test.py` — CRF test eval (Viterbi + emission-only + bp P/R).
- `src/python/crf/eval_hmm.py` — Li–Stephens HMM baseline scorer (per-window,
  sweeps p_stay/weight).
- `src/python/crf/eval_stratified.py` — compare arms by true-breakpoint band.
- `src/python/crf/metrics.py` — breakpoint precision/recall (±tol).
- `src/python/crf/train_crf.py` — shared `FounderPathEncoder` (don't rewrite);
  `--window-c` lives here.

## Next steps (prioritized — user-approved direction is diploid)
1. **Diploid HMM baseline** — wrap `hmm/hmm_impute.py:diploid_hmm` like
   `eval_hmm.py` (per-window, batched, same test split, with `homo_penalty`) to
   get a CRF-vs-HMM head-to-head in the diploid setting. (Mirrors what made the
   haploid story land.)
2. **Window 1024** — user wants to test whether larger windows help diploid
   phasing: regenerate sim with `--sites 1024` and retrain.
3. **Per-coverage stratification** — does the CRF edge grow at low depth?
4. Deferred: ≥3-seed / converged / full-data headline (PLAN §8); relatedness
   encoder (E5); promote to real diploid maize.

## Gotchas
- The OTHER maize file `…/fullMaizeDataset.npy` (no `_all_diploid`) is BROKEN —
  all 25 cols are read counts, no label column. Use the `_all_diploid` file.
- `pixi` binary at `~/.pixi/bin/pixi`; imports use `from python.<mod>` (PYTHONPATH
  set by pixi to `src/`).
- bf16 required (fp16 overflows the CRF partition logsumexp); lr 1e-4 + warmup 500.
