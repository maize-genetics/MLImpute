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
| E1  | CRF (full)  |  ~5M   | —           | —          | —           | —              | —             | —          |
| E1  | CRF (no-transition) | ~5M | —      | —          | —           | —              | —             | —          |

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
