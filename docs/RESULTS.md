# GRITS-CRF — Results leaderboard

Running record of every experiment. **Append rows; never overwrite prior rows**
(`CLAUDE.md`). Each headline comparison vs the HMM should report mean ± sd over ≥3
seeds; single-run diagnostics are labeled as such.

Columns follow the `docs/PLAN.md` §7 template. `—` = not yet measured (most depend on
`metrics.py` / `baselines.py`, still to be built — see PLAN §2.2).

| Exp | Model / arm | Params | Founder acc | Switch err | Breakpt P/R | In-SV false bp | SNP acc (MAF) | Mem / it/s |
|-----|-------------|-------:|------------:|-----------:|-------------|----------------|---------------|------------|
| E0  | PHG-HMM     |   n/a  | —           | —          | —           | —              | —             | —          |
| E0  | GRITS       |  ~43M  | —           | —          | —           | —              | —             | —          |
| E1-probe | CRF, window-avg emis (fp16, lr 3e-4) | 5.07M | 0.419 | — | — | — | — | ~5.1 it/s |
| E1-probe | CRF, time-local emis (fp16, lr 3e-4) | 5.07M | 0.04–0.21† | — | — | — | — | ~5.1 it/s |
| **E1-probe** | **CRF, time-local emis (bf16, lr 1e-4, warmup 500)** | 5.07M | **0.990** | — | — | — | — | ~5.1 it/s |
| **E1-probe-coal** | **CRF, coalescent mini-hap SFS (bf16, lr 1e-4, warmup 500)** | 5.07M | **0.863** | — | — | — | — | ~5.0 it/s |
| **E2-probe-coal** | **CRF, coalescent + variable recomb (span 100)** | 5.07M | **0.833**‡ | — | Spearman(c_t,rate) **+0.002** | — | — | ~5.0 it/s |
| **E2b-probe-coal** | **+ recomb_head aux loss (corr, weight 5)** | 5.07M | **0.835**§ | — | val recomb_corr **+0.005** (no change) | — | — | ~5.0 it/s |
| **E2b-probe-coal** | **+ recomb_head aux loss (corr, weight 200)** | 5.07M | **0.838** | — | Spearman(c_t,rate) **−0.001** (no change) | — | — | ~5.0 it/s |
| E1  | CRF (full)  |  ~5M   | —           | —          | —           | —              | —             | —          |
| E1  | CRF (no-transition) | ~5M | —      | —          | —           | —              | —             | —          |

‡ best epoch (epoch 1) val acc; the run became unstable and diverged at epoch 2
(train loss bouncing 11↔58, val acc → 0.04). Reported from the epoch-1 checkpoint.

§ best epoch (epoch 2) val acc; same alternating-epoch instability (val acc
0.815 / 0.10 / 0.835 / 0.04 at val steps 1–4). Reported from the best epoch.

† diverged mid-training (fp16 overflow in the CRF partition `logsumexp`); val acc
decayed to random by epoch 1. Listed to document the failure mode, not as a result.

---

## E1-probe — Haploid encoder validation on the synthetic allele-sharing sim (2026-06-23)

**Not a scored E1 result.** A controlled architecture probe on
`crf/simulate_alleles.py` (binary allele-match features, inbred so H1≡H2, predicting
H1), *not* the `cross/` data source and *not* compared to a baseline. Purpose: isolate
the emission vs. transition mechanism. Founder acc = validation accuracy (Viterbi vs.
true H1 path), not the §4.3 panel-masked metric.

- **Data:** `data/training/sim_alleles.npy` — 100k windows × 512 sites × 26 cols
  (K=24 founders + H1 + H2). `allele-sharing=0.2`, `bad-frac=0.05`,
  crossovers 2–10/window, `inbreeding=1.0`. True-founder allele match 0.958 (the
  per-site oracle ceiling). Split 80k/10k/10k.
- **Model:** `GRITSCRFHaploid`, d=256, L=6, 5,068,806 params, K=25 states. 2 epochs.
- **Runs (GPU 1):**

  | emission key | precision | LR | warmup | val acc | val loss | note |
  |---|---|---|---:|---:|---:|---|
  | window-averaged | 16-mixed | 3e-4 | 0 | 0.419 | 50.2 | gate collapses to ~0.12; CRF on smoothing prior |
  | time-local | 16-mixed | 3e-4 | 0 | 0.04–0.21 | 46.9 | gate→1.0, loss hits 2.9 then **diverges** to 61 |
  | **time-local** | **bf16-mixed** | **1e-4** | **500** | **0.990** | **9.47** | gate→1.0, stable; `stay_bonus`/`c` now learn |

- **Takeaways:**
  1. **Emissions must be time-local.** A window-averaged founder key cannot localize
     the active founder when the path switches within a window → emission gate closes
     → 0.42. Per-site key opens the gate and reaches 0.99.
  2. **bf16 is required.** fp16 overflows the partition `logsumexp` once emissions are
     active; bf16 + warmup + lower LR fixes it.
  3. **Transition cost `c` collapsed to a constant (≈5)** because the sim's
     recombination is spatially uniform — motivates E2 (variable rate map).
- **Repro:**
  ```bash
  pixi run -- python src/python/crf/simulate_alleles.py --workdir /workdir/esb33
  CUDA_VISIBLE_DEVICES=1 pixi run --environment gpu -- python \
    src/python/crf/train_haploid.py --data /workdir/esb33/data/training/sim_alleles.npy \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed --max-epochs 2
  ```
- **Checkpoint:** `<workdir>/checkpoints/e1-haploid/e1-epoch=00-val/loss=9.466.ckpt`
- **Commit:** `ce381d5` (branch `crf-relatedness`).

---

## E1-probe-coal — Coalescent mini-haplotype (SFS) sim (2026-06-23)

**Not a scored E1 result.** Same encoder probe as E1-probe, but on the harder
**coalescent mini-haplotype** features (`--sharing-model coalescent`): each site is a
150bp read of `read_snps=8` biallelic SNPs (per-SNP derived freq ~ Beta(0.3, 1) SFS),
and a founder "matches" only on an exact full-read agreement. Founders are mosaics of
`ancestors=6` ancestral lineages, so they share haplotype tracts by descent — real
LD + founder relatedness rather than per-site independent sharing.

- **Data:** `data/training/sim_coal.npy` — 100k × 512 × 26 (K=24 + H1 + H2),
  `inbreeding=1.0`, crossovers 2–10/window. True-founder match **0.964** (per-site
  oracle ceiling); **match LD lag-1 = 0.436** (vs ≈0 for the independent sim) —
  confirms the reads form allele-sharing tracts. Split 80k/10k/10k.
- **Model:** `GRITSCRFHaploid`, d=256, L=6, 5.07M params, K=25 states. 3 epochs,
  time-local emis, bf16-mixed, lr 1e-4, warmup 500.
- **Result:** val acc **0.863** (epoch 2, best `val/loss=11.169`). Below the 0.99 of
  the independent sim and the 0.964 per-site oracle — expected: relatedness makes
  multiple founders match the same read, so the true founder is no longer point-wise
  identifiable and the model must lean on temporal continuity. This is the intended
  harder, more realistic setting for the relatedness work.
- **Fix applied:** `simulate_alleles.py` was missing the `--read-snps` CLI flag and
  the `main()→simulate()` call dropped the `read_snps` positional, shifting
  `error_block` into the read-length slot (a float → `rng.beta` crash). Both fixed.
- **Repro:**
  ```bash
  pixi run -- python src/python/crf/simulate_alleles.py --workdir /workdir/esb33 \
    --windows 100000 --sharing-model coalescent --out sim_coal.npy
  CUDA_VISIBLE_DEVICES=0 pixi run --environment gpu -- python \
    src/python/crf/train_haploid.py --data /workdir/esb33/data/training/sim_coal.npy \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed --max-epochs 3
  ```
- **Checkpoint:** `<workdir>/checkpoints/e1-haploid/e1-epoch=02-val/loss=11.169.ckpt`
- **Branch:** `crf-relatedness` (sim fix + run uncommitted at time of writing).

---

## E2-probe-coal — Does inferred c_t track the hidden recomb rate? (2026-06-24)

**Negative result.** Coalescent mini-haplotype features **+** the E2 hidden variable
recombination-rate map (`--recomb-span 100`), to test whether the encoder infers
local recombination from the allele-match patterns (LD breakdown) and feeds it into
the transition cost `c_t`. Evaluated with `eval_recomb.py` on the epoch-1 checkpoint.

- **Data:** `data/training/sim_coal_e2.npy` — 100k × 512 × 27 (K=24 + H1 + H2 +
  hidden rate). Hidden rate 1–100 (mean 21.5); `corr(rate, switch)=0.10` (breakpoints
  do follow the rate); **match-persist hot/cold = 0.734 / 0.757** — the rate *is*
  observable in the features, but the hot–cold gap is small (0.023).
- **Train:** same recipe as E1-probe-coal. val acc 0.833 at epoch 1 (best), then the
  run diverged at epoch 2 (instability, not fp16 — this is bf16).
- **Eval (`eval_recomb.py`, 2k held-out windows):**
  - `c_t` **collapsed to a near-constant**: mean 4.376, sd 0.012 (range 4.33–4.41).
  - `Spearman(c_t, hidden rate)  = +0.002` (expected strong negative) → **no tracking**.
  - `Spearman(c_t, true switch)  = -0.002` → does not localize breakpoints either.
  - cold-decile vs hot-decile mean `c_t`: 4.376 vs 4.376 (gap 0.000).
- **Takeaway:** the encoder does **not** infer local recombination rate; the
  transition cost collapses to a global constant (~4.4), the same failure mode flagged
  in E1-probe (`c≈5`). A constant `c` is sufficient to reach 0.83 founder acc, so there
  is no training pressure to make `c` input-dependent. Making `c_t` track the hidden
  rate is the open E2 problem — candidate levers: stronger hot/cold feature contrast
  (larger `--recomb-span` / smaller `--recomb-tile`), an explicit `recomb_head`
  auxiliary loss, longer/stabilized training, or a richer transition parameterization.
- **Repro:**
  ```bash
  pixi run -- python src/python/crf/simulate_alleles.py --workdir /workdir/esb33 \
    --windows 100000 --sharing-model coalescent --recomb-span 100 --out sim_coal_e2.npy
  CUDA_VISIBLE_DEVICES=0 pixi run --environment gpu -- python \
    src/python/crf/train_haploid.py --data /workdir/esb33/data/training/sim_coal_e2.npy \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed --max-epochs 3
  CUDA_VISIBLE_DEVICES=0 pixi run --environment gpu -- python \
    src/python/crf/eval_recomb.py --ckpt <epoch-1 ckpt> \
    --data /workdir/esb33/data/training/sim_coal_e2.npy
  ```
- **Checkpoint:** `<workdir>/checkpoints/e1-haploid/e1-epoch=01-val/loss=15.044.ckpt`
- **Branch:** `crf-relatedness` (uncommitted at time of writing).

---

## E2b-probe-coal — recomb_head auxiliary loss on c_t (2026-06-24)

**Negative result.** Direct follow-up to E2-probe-coal: add an explicit auxiliary
loss that pushes the inferred transition cost `c_t` to anti-correlate with the
hidden recombination rate, testing whether *supervising the ranking* (hot ⇒ cheap
to switch) can break the `c_t`-collapses-to-a-constant failure mode. It does not.

- **Mechanism:** `_window_corr(c, log_rate)` = mean per-window Pearson corr between
  `c_t` and the (log) hidden rate, added to the loss as `recomb_aux_weight * corr`
  (minimizing it should drive `corr → −1`). Scale/shift-invariant in `c`, so it
  supervises only the *shape*, never `c`'s absolute scale. The hidden rate is an
  **auxiliary target only** — `PreWindowedHaploidDataset` exposes it as `log_rate`
  (col K+2 of the K+3 layout) and it is **never** fed into the forward pass.
- **Data:** `data/training/sim_coal_e2.npy` (same as E2-probe-coal).
- **Train:** E1-probe-coal recipe + `--recomb-aux-weight 5.0`. 3 epochs, bf16,
  lr 1e-4, warmup 500. (`version_12`; `version_11` was a smoke run on the small file.)
- **Result (TensorBoard `version_12`):**
  - `val/recomb_corr`: 0.0014 → 0.0016 → 0.0055 → 0.0017 — **never moves off zero.**
  - `train/recomb_corr`: bounces −0.035…+0.066 with no downward trend — the
    minimized term exerts no effective pressure.
  - `train/recomb` (mean `c`): drifts 4.2 → 6.9 (only the *scale* moves; the
    spatial structure does not).
  - `val/acc`: 0.815 / 0.10 / 0.835 / 0.04 — same alternating-epoch instability as
    E2-probe-coal (best 0.835 at epoch 2).
- **Why it failed (hypothesis):** the aux term is numerically negligible. With
  `corr ≈ O(0.01)` and weight 5, it contributes ≈0.05 to a CRF loss of ≈14 — three
  orders of magnitude too small to reshape `c_t`. A constant `c` remains a loss
  minimum the optimizer is happy to sit in. Next: a far larger weight (100–500)
  and/or a gradient-magnitude-matched formulation, plus stabilization for the
  alternating-epoch divergence.
- **Repro:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 pixi run --environment gpu -- python \
    src/python/crf/train_haploid.py --data /workdir/esb33/data/training/sim_coal_e2.npy \
    --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
    --max-epochs 3 --recomb-aux-weight 5.0
  ```
- **Checkpoint:** `<workdir>/checkpoints/e1-haploid/e1-epoch=02-val/...ckpt`
- **Branch:** `crf-relatedness`.

### E2b retry — larger aux weight (200) + stability (2026-06-24)

Follow-up to the weight-5 null: a smoke sweep (`sweep_recomb_aux.py`, 300 steps,
3k windows) showed the aux loss *can* break the constant-`c` collapse — `c`'s
spatial sd rose with weight (0.09→0.51 at w=500) — but `recomb_corr` saturated at
≈−0.02 by weight 50, i.e. the added variance is not aligned with the rate. Full
run at **weight 200** (`version_13`, 3 epochs, same recipe):

- **val/acc:** 0.288 → 0.834 → 0.838 — **stable** this time (the higher weight
  removed the alternating-epoch divergence seen at weight 5).
- **val/recomb_corr:** 0.0008 / 0.0033 / 0.0001 — still pinned at ~0.
- **eval_recomb.py (2k held-out):** `c_t` mean 6.187, **sd 0.016** (collapsed
  again on held-out data), **Spearman(c_t, rate) = −0.0005**, cold−hot decile gap
  0.000. The training-time `c` variance was overfitting, not rate-tracking.
- **Verdict:** a corr-based auxiliary loss does **not** make `c_t` track the
  hidden rate at any weight tried (5–500). Raising the weight only adds
  unaligned variance (and, usefully, stabilizes training). The bottleneck is
  upstream — the hot/cold feature contrast is too weak (match-persist gap 0.023,
  E2-probe-coal). Next levers should target the *features*: larger `--recomb-span`
  / smaller `--recomb-tile` for sharper hot/cold contrast, or a direct per-site
  rate-regression head supervised against the hidden track, before retrying the
  transition-cost coupling.
- **Checkpoint:** `<workdir>/checkpoints/e1-haploid/e1-epoch=02-val/loss=13.514.ckpt`
- **Branch:** `crf-relatedness`.
