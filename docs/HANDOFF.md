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
- `src/python/crf/eval_hmm.py` — haploid Li–Stephens HMM baseline scorer
  (per-window, sweeps p_stay/weight).
- `src/python/crf/eval_diploid_hmm.py` — diploid Li–Stephens baseline (factored
  per-chromosome transition; sweeps p_stay×weight×homo_penalty).
- `src/python/crf/eval_test_diploid.py` — held-out evaluator for
  `GRITSCRFDiploid` checkpoints (pair_acc / sorted hap_acc / homo-pred fraction).
- `src/python/crf/eval_stratified.py` — compare arms by true-breakpoint band.
- `src/python/crf/metrics.py` — breakpoint precision/recall (±tol).
- `src/python/crf/train_crf.py` — shared `FounderPathEncoder` (don't rewrite);
  `--window-c` lives here.

### 3. Diploid HMM baseline + d128/L4 capacity — DONE (E4-probe-baseline, 2026-06-24)
New scorers `eval_diploid_hmm.py` (Li–Stephens baseline, factored per-chromosome
transition, sweeps p_stay×weight×homo_penalty) and `eval_test_diploid.py`
(held-out `GRITSCRFDiploid` evaluator). Same test split of `sim_diploid_512.npy`.

| arm | params | test pair_acc | test hap_acc |
|---|---:|---:|---:|
| CRF full d256/L6 (hp=3) | 5.07M | 0.614 | 0.743 |
| **CRF d128/L4 (hp=3)** | 0.88M | **0.618** | **0.746** |
| **Diploid HMM (best)** | ~0 | 0.552 | 0.700 |

CRF beats the diploid HMM by **+6.5 pair / +4.6 hap**; the 0.88M model ties the
5.07M one (capacity is not the bottleneck — same as haploid). Best HMM:
p_stay=0.99, w=0.5, hp=0.5. CRF predicted-homozygous fraction 0.0005 (well-
calibrated het prior). See RESULTS.md "E4-probe-baseline".

## Next steps (prioritized — user-approved direction is diploid)
1. **Window 1024** — user wants to test whether larger windows help diploid
   phasing: regenerate sim with `--sites 1024` and retrain.
2. **Per-coverage stratification** — does the CRF edge grow at low depth?
3. Deferred: ≥3-seed / converged / full-data headline (PLAN §8); relatedness
   encoder (E5); promote to real diploid maize.

### Future ideas (user-flagged 2026-06-24, also in Claude memory)
- **Masked-site accuracy (SNP-like metric):** mask a fraction of test sites and
  check whether the decoded path recovers the correct founder there —
  complements haplotype-path accuracy. Needs the sim to mark masked/bad sites
  (extra label column) and the eval to score only on those positions.
- **Sub-panel / bi-parental sim:** generate *test* sets where each sample
  descends from only a subset of founders (simplest: bi-parental, K=2). Train on
  the full panel; keep the feature matrix K-wide (other founders just have zero
  coverage) so the trained model evaluates unchanged. Add `--founders-subset` to
  `simulate_alleles.py`.

## Gotchas
- The OTHER maize file `…/fullMaizeDataset.npy` (no `_all_diploid`) is BROKEN —
  all 25 cols are read counts, no label column. Use the `_all_diploid` file.
- `pixi` binary at `~/.pixi/bin/pixi`; imports use `from python.<mod>` (PYTHONPATH
  set by pixi to `src/`).
- bf16 required (fp16 overflows the CRF partition logsumexp); lr 1e-4 + warmup 500.
- `pixi run --environment gpu` HANGS on this cluster's NFS (blocks in `cl_syn` on
  env resolution, never reaches Python). Workaround: call the env Python directly:
  `PYTHONPATH=src CUDA_VISIBLE_DEVICES=N .pixi/envs/gpu/bin/python <script>`. For
  scripts that import scipy (e.g. `eval_test_diploid.py`) also set
  `LD_LIBRARY_PATH=.pixi/envs/gpu/lib` (bare Python misses the newer libstdc++).
