# GRITS-CRF — Implementation Plan & Experiment Series

> **Audience:** Claude Code. This is an execution spec. Build incrementally, one
> experiment at a time, and do not start an experiment until the previous one's
> acceptance criteria are met and logged. The companion file `grits_crf.py` is a
> verified, runnable skeleton (loss 229→33 on synthetic mosaics) — use it as the
> starting model module, not as final code.

---

## 1. Objective

Impute **founder haplotype paths** (the diploid mosaic of which ancestral founder
each haplotype descends from at each position), **not SNP dosages**. Beat a
well-tuned Li–Stephens **HMM** baseline on low-coverage (0.01–2×), structurally
variable, imperfectly-nominated plant data.

**Core hypothesis to test:** the HMM wins because of its structured Markov-mosaic
prior, not its capacity. Keeping a CRF+Viterbi backbone but making its emission and
transition potentials neural and input-conditional will match/beat the HMM at
~10× fewer parameters than the current 43M GRITS model, and will turn the HMM's
three weaknesses (recombination-rate heterogeneity, SV/bad-data blocks, no global
conditioning) into learnable behavior.

---

## 2. Wire-up TODOs (fill before running — these are environment-specific)

These are placeholders Claude Code must connect to the real environment. Where the
real API is unknown, implement against the **interface contract in §4** and leave a
clearly marked `# TODO(wire):` shim.

- `TODO(wire)`: Simulator entry point that produces (panel + reads → PS4G matrix +
  true path). Path/CLI/Python API unknown. Must expose the nomination step so it can
  be run *inside* the data pipeline.
- `TODO(wire)`: ropebwt3 → PS4G construction (read support per nominated founder per
  position). Produces the input tensor `X`.
- `TODO(wire)`: PHG-HMM baseline invocation + output parser (to score on identical
  held-out sims). This is the number to beat.
- `TODO(wire)`: GRITS (ModernBERT+GRU) checkpoint/runner, for re-scoring on the same
  held-out set and for the parameter-count comparison.
- `TODO(wire)`: compute budget (GPU model/VRAM) — needed to size `d`/`L`/window and
  to choose Transformer vs Mamba2.

---

## 3. Proposed repo layout

```
grits_crf/
  model.py            # from grits_crf.py: FounderPathEncoder, NeuralCRF, diploid_*
  data.py             # Dataset/loader over simulator output (interface in §4)
  metrics.py          # founder_acc, switch_error, breakpoint_localization, snp_acc
  baselines.py        # run_phg_hmm(...), score_grits(...) wrappers (TODO wire)
  train.py            # config-driven training loop, checkpoint, logging
  decode.py           # chromosome-wide Viterbi (founder + pair-state)
  configs/            # one yaml per experiment (E0..E7)
  experiments/        # scripts + results tables per experiment
  tests/              # shape/grad/decode unit tests + golden-path overfit test
  RESULTS.md          # the running leaderboard (template in §7)
```

---

## 4. Interface contracts (implement to these exactly)

### 4.1 Data sample
A single sample is one chromosome (or window) of one diploid individual:

```python
# data.py
@dataclass
class Sample:
    X: Tensor            # [T, K] float, read support of nominated founder k at site t
    tags: Tensor         # [T]    long, true founder index (single-hap) OR
                         # [T]    long, true unordered-pair-state index (diploid)
    founder_mask: Tensor # [K]    {0,1}, 1 = founder present in this sample's panel
    dbp: Tensor          # [T]    bp (or cM) gap to next informative site, >0
    panel_id: int        # which nominated panel (for transfer splits)
    meta: dict           # {coverage, n_true_switches, sv_blocks:[(start,end)], ...}
```

- **Imperfect nomination is mandatory:** run the simulator's nomination so the panel
  is genuinely the nominated top-K. When the true founder is absent, set `tags` to the
  **best available** founder in the panel (decided: no off-panel state).
- Variable K across samples → pad to max K in a batch and use `founder_mask`.
- Splits: train / val / **held-out test** with *unseen panels* reserved for E5.

### 4.2 Model (already in `grits_crf.py`)

```python
enc = FounderPathEncoder(d_model, n_layers, ext_dim=0)   # index-agnostic over founders
emis, gate, c = enc(X, founder_mask, dbp, ext_emb=None)  # emis[B,T,K], gate[B,T], c[B,T]
crf = NeuralCRF()
loss = crf.nll(emis, c, founder_mask, tags)              # CRF NLL of the true path
path = crf.viterbi(emis, c, founder_mask)                # decode
# diploid: diploid_pair_index(K), diploid_emissions(...), diploid_transition(...)
```

### 4.3 Metrics (`metrics.py`)
- `founder_accuracy(pred, tags, mask)` — per-site, panel-masked.
- `switch_error_rate(pred, tags)` — phasing/switch errors per informative interval.
- `breakpoint_localization(pred, tags, tol_bp)` — precision/recall of breakpoints
  within `tol_bp`; also report **false breakpoints inside `meta.sv_blocks`** (key for E3).
- `snp_accuracy_from_path(pred, panel)` — reconstruct SNP dosages from the path and
  score MAF-binned, for apples-to-apples vs SNP imputers.
- `param_count(model)` and a `(window_len, K, peak_mem, throughput)` profiler.

---

## 5. Experiment series

Run in order. Each experiment = one config + one script + one row in `RESULTS.md`.
Every experiment must include the **ablation** named in its acceptance criteria.

### E0 — Harness & baselines (no modeling)
- **Objective:** make every later number interpretable.
- **Implement:** `data.py` over simulator output; `metrics.py`; `baselines.py`
  wrappers to run/score PHG-HMM and re-score GRITS on one shared held-out sim set.
- **Run:** score HMM and GRITS; write the first `RESULTS.md` rows.
- **Acceptance:** identical held-out set scored by both baselines on all §4.3 metrics;
  GRITS param count reproduced (~43M); a golden-path **overfit test** (model can drive
  CRF NLL→~0 on a single batch) passes in `tests/`.
- **Decision informed:** baseline bar + measurement validity.

### E1 — Structured-layer ablation (the central test)
- **Objective:** does the CRF prior close the gap to the HMM at small parameter count?
- **Implement:** single-haplotype training with `FounderPathEncoder + NeuralCRF`
  (drop any GRU). Train on **un-split** chromosomes (one recombination regime).
  Ablation arm: **emission-only / no transition** (set `c→∞` so switches are free →
  per-site argmax) to isolate the CRF's contribution.
- **Run:** `configs/e1.yaml` at `d=256, L=6` (~5M params; see §6).
- **Acceptance:** (a) founder accuracy ≥ HMM on held-out; (b) params ≤ ~6M
  (≈8× under GRITS); (c) full CRF beats the no-transition ablation by a clear margin.
- **Decision informed:** is "neuralize the HMM" the right direction at all?

### E2 — Recombination-rate heterogeneity
- **Objective:** replace hand-split chromosome regions with learned `c_t`.
- **Implement:** train across simulated rate regimes spanning ~2 orders of magnitude;
  feed `dbp` into the recomb head. Ablation arm: **constant transition** (learned scalar,
  no per-site `c_t`).
- **Run:** sweep crossover frequency; plot accuracy-vs-crossover-frequency.
- **Acceptance:** learned `c_t` flattens the accuracy-vs-crossover curve relative to the
  constant-transition arm; matches an HMM given the *true* per-site map without being told it.
- **Decision informed:** is learned `c_t` sufficient, or is an explicit map covariate needed?

### E3 — SV / bad-nomination robustness + reliability gate
- **Objective:** stop spurious breakpoints inside correlated-bad-data blocks.
- **Implement:** inject SV/indel blocks and bad-nomination blocks in sim (record spans in
  `meta.sv_blocks`); keep `tags` constant through them. Ablation arms: **gate off**
  (`g≡1`) and **gate without budget reg** (test for collapse).
- **Run:** `configs/e3.yaml`; report false-breakpoint rate *inside* injected blocks.
- **Acceptance:** gate reduces in-block false breakpoints vs gate-off **without** raising
  real-breakpoint error; gate does not collapse to ~0 (budget reg holds it near 1 elsewhere);
  no identifiability blow-up between low `g` and high `c` (inspect their correlation).
- **Decision informed:** is the gate+`c_t` mechanism well-posed?

### E4 — Diploid pair-state
- **Objective:** joint, phased two-path decode.
- **Implement:** `decode.py` chromosome-wide Viterbi over unordered pairs
  (`diploid_pair_index`, `diploid_emissions`, `diploid_transition`, one-switch rule);
  switch the matrixized transition to the **matrix-free stay/switch recursion** for the
  pair-state space (do not materialize `[B,T,P,P]`). Ablation arm: **two independent
  haplotype decodes + post-hoc pairing**.
- **Run:** K=24; profile decode cost `O(T·P²)` and memory.
- **Acceptance:** lower switch-error than the independent-decode arm; tractable at K=24
  and target chromosome length; zero added parameters vs E1 (verify).
- **Decision informed:** pair-state vs per-hap+coupling; tractability ceiling on K.

### E5 — Variable / imperfect nomination + transfer
- **Objective:** generalize to nominated panels unseen in training (index-agnostic claim).
- **Implement:** vary the nominated panel per sample; train across many panels; reserve
  **unseen panels** for test. Optionally add a frozen external assembly embedding via
  `ext_dim` (e.g. PlantCAD2) and measure its marginal value.
- **Run:** `configs/e5.yaml`; report accuracy on seen vs unseen panels.
- **Acceptance:** accuracy on unseen panels within a small, pre-registered margin of seen
  panels (no catastrophic transfer drop); external embedding either helps or is dropped.
- **Decision informed:** does it transfer, or is per-panel calibration unavoidable?

### E6 — Long-context encoder (Mamba2) + scaling
- **Objective:** full-chromosome decode within compute budget.
- **Implement:** swap the Transformer position-encoder for **Mamba2** behind the same
  interface; keep CRF unchanged.
- **Run:** sweep window length; profile peak memory & throughput; confirm **param count
  is invariant to T and K** (assert in `tests/`).
- **Acceptance:** Mamba2 matches Transformer accuracy on E1's task at lower memory for
  long windows; full target chromosome decodes within the §2 GPU budget.
- **Decision informed:** encoder choice and maximum single-pass window.

### E7 — Global / unlinked-chromosome conditioning (optional, last)
- **Objective:** use genome-wide founder presence to disambiguate locally ambiguous calls.
- **Implement:** pool a genome-wide founder-presence/proportion vector and inject it into
  the emission/recomb heads. Ablation arm: **off**.
- **Run:** `configs/e7.yaml`.
- **Acceptance:** improves accuracy specifically in low-confidence regions (slice by `gate`
  / posterior entropy) without hurting elsewhere; otherwise drop it.
- **Decision informed:** keep or discard global conditioning.

---

## 6. Default config & parameter budget

Trainable params ≈ `(5 + 12·L)·d² + (10 + 13·L)·d + 6` (exact). Reference points:
`d=96,L=2 → 271K`; `d=128,L=4 → 876K`; **`d=256,L=6 → 5.07M` (default for E1+)**;
`d=384,L=12 → 22M`. **Independent of T (sites) and K (founders); diploid adds 0 params.**
Start at the 5M config; only scale `d`/`L` if E1 underfits.

---

## 7. RESULTS.md leaderboard template

| Exp | Model / arm | Params | Founder acc | Switch err | Breakpt P/R | In-SV false bp | SNP acc (MAF) | Mem / it/s |
|-----|-------------|-------:|------------:|-----------:|-------------|----------------|---------------|------------|
| E0  | PHG-HMM     |   n/a  |             |            |             |                |               |            |
| E0  | GRITS       |  ~43M  |             |            |             |                |               |            |
| E1  | CRF (full)  |  ~5M   |             |            |             |                |               |            |
| E1  | CRF (no-transition) | ~5M |        |            |             |                |               |            |

---

## 8. Engineering conventions
- Config-driven (one yaml per experiment under `configs/`); log seed, git SHA, config.
- Fixed seeds; report mean ± sd over ≥3 seeds for any headline comparison vs HMM.
- `tests/` must include: shape tests, a gradient/finite-difference check on the CRF NLL,
  a Viterbi-vs-brute-force agreement test on a tiny state space, and the golden-path
  overfit test from E0.
- Every experiment appends to `RESULTS.md`; never overwrite prior rows.
- Keep the model module's public interface (§4.2) stable across experiments.

## 9. Guardrails / non-goals
- Do **not** predict SNPs as the training target; the target is the path. (SNP accuracy
  is a *reconstructed* downstream metric only.)
- Do **not** add a per-founder-index parameter anywhere — it breaks variable-panel transfer.
- Do **not** materialize `[B,T,P,P]` transitions in the diploid decode; use the
  matrix-free recursion.
- Do **not** tune against the test split or against the real-data eval; reserve those.

## 10. Settled vs. open decisions
- **Settled:** path restricted to nominated top-K; no off-panel state; truth = best
  available founder; nomination simulated inside the pipeline; `c_t`/`g_t` learned
  end-to-end (never directly supervised).
- **Open (resolve via experiments):** Transformer vs Mamba2 (E6); pair-state vs
  per-hap+coupling (E4); whether global conditioning earns its place (E7); whether an
  explicit recombination-map covariate is needed beyond `c_t` (E2); value of an external
  founder embedding (E5).
