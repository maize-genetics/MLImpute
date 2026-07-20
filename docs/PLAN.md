# GRITS-CRF — Implementation Plan & Experiment Series

> **Audience:** Claude Code. This is an execution spec for the diploid neural-CRF
> founder-path imputer on the `crf-relatedness` branch. **Last reconciled with the
> code: 2026-06-23** (see §0 Progress log). Build incrementally, one experiment at a
> time, and do not start an experiment until the previous one's acceptance criteria
> are met and logged in `RESULTS.md`.
>
> **What changed since the original draft:** the simulator and PS4G→matrix wiring
> are no longer stubs — they are implemented in `src/python/cross/`. The current
> `train_crf.py` is wired diploid-only, but the **first training milestone is
> haploid**: get the system trained end-to-end on single-haplotype data first, while
> keeping the architecture **diploid-capable from the start** so no rework is needed
> to promote to pair-state. This is cheap because the encoder is founder-level and
> the CRF is state-count-agnostic (see §4.2). The E0 measurement harness
> (`metrics.py`, baseline scoring, `RESULTS.md`, `tests/`) is still **not built** and
> remains the gating first step.

> **Architecture invariant (hold this across all experiments):** one shared
> `FounderPathEncoder` produces founder-level emissions `emis_f [B,T,K]`. Haploid
> and diploid differ **only in the state layer fed to the same `NeuralCRF`** —
> haploid runs over `K` founder states with `nsw[i,j] = (i!=j)`; diploid runs over
> `P=(K+1)(K+2)/2` pair states with `nsw =` min-over-orderings. Never fork the
> encoder or add a per-founder-index parameter; the haploid milestone must not close
> the door on the diploid path.

---

## 0. Progress log

### 2026-06-23 — Haploid encoder architecture validated on a synthetic allele-sharing sim

A controlled probe of the E1 encoder, ahead of scoring on the `cross/` pipeline data.
Built a well-posed standalone simulator and used it to isolate the emission vs.
transition mechanism. **This is pre-E1 architecture validation, not E1 completion** —
no HMM baseline, no `cross/` data, no `metrics.py` yet, so E1's acceptance criteria
(§5 E1) remain open.

- **New simulator `crf/simulate_alleles.py`** (separate from the `cross/` PS4G pipeline;
  the `cross/` set remains the data source of record for scored experiments). Emits
  `[windows, sites, K+2]` int8: K binary allele-match features + H1 + H2 founder labels.
  Each window is a piecewise-constant founder path observed only through a noisy
  allele-match pattern; the true founder matches the sample allele on `1-bad_frac` of
  sites. Controls: `--allele-sharing`, `--inbreeding` (F), `--bad-frac`,
  `--min/--max-crossovers`. Default 100k×512×26 inbred set written to
  `data/training/sim_alleles.npy`.
- **`train_haploid.py` wired to it.** `PreWindowedHaploidDataset` now accepts the
  `(N,T,K+2)` diploid layout and predicts **H1** (col K; H2 ignored — valid because the
  sim is inbred, H1≡H2); `make_splits` validates/reports the layout.
- **Key architecture finding — emissions must be time-local.** The original
  `FounderPathEncoder` emission key is averaged over the window
  (`e = cells.mean(dim=1)`), which cannot localize *which* founder is active at site
  `t` once the path switches within a window (sim averages ~6 crossovers/512). Result:
  the emission gate collapses to ~0.12 and the CRF falls back on its smoothing prior →
  **val acc 0.42**. Added a flagged per-site emission key
  (`time_local_emis`, `einsum("btd,btkd->btk", H, cells)`): the gate snaps to 1.0 and
  2-epoch **val acc → 0.99**. The binary 0/1 allele-match representation was kept (it is
  the right signal); the fix is purely in how the per-site founder score is formed.
- **Training-stability fix.** With emissions active, `precision="16-mixed"` overflowed
  the CRF partition `logsumexp` and training diverged (loss 2.9→61, val acc→random).
  Switched default to **`bf16-mixed`**, added `--warmup-steps` (linear) and lowered LR
  to 1e-4 → stable, no divergence. New `train_haploid.py` flags: `--time-local-emis`,
  `--warmup-steps`, `--precision`. The shared `FounderPathEncoder` change is **off by
  default** so the maize/`cross/` path is unchanged.
- Numbers logged in `docs/RESULTS.md` (E1-probe rows).

**Next (Step 2 → folds into E2):** add a variable per-site recombination-rate map to
`simulate_alleles.py` spanning ~100× (hot/cold regions), emit it as a `dbp`-style
covariate track, and feed it into `recomb_head` so the transition cost `c_t` — and
thus the transition↔emission balance — adapts to local LD instead of collapsing to a
constant (`c≈5`, as observed). See §5 E2.

---

## 1. Objective

Impute **founder haplotype paths** (the diploid mosaic of which ancestral founder
each haplotype descends from at each position), **not SNP dosages**. Beat a
well-tuned Li–Stephens **HMM** baseline on low-coverage (0.01–2×), structurally
variable, imperfectly-nominated plant data.

**Core hypothesis to test:** the HMM wins because of its structured Markov-mosaic
prior, not its capacity. Keeping a CRF+Viterbi backbone but making its emission and
transition potentials neural and input-conditional will match/beat the HMM at
~10× fewer parameters than the current ~43M GRITS model, and will turn the HMM's
three weaknesses (recombination-rate heterogeneity, SV/bad-data blocks, no global
conditioning) into learnable behavior.

---

## 2. What is built vs. still open

### 2.1 Built (do not re-stub these)

| Area | Where | Notes |
|------|-------|-------|
| Recombination-mosaic simulation | `cross/pick_crossovers.py` | Crossover spacing model (`min_spacing`/`max_spacing`), tail-safe. |
| Synthetic allele-sharing sim | `crf/simulate_alleles.py` | Standalone well-posed probe (binary allele-match features + H1/H2 labels); used for E1 architecture validation, **not** the scored data source. Uniform recombination so far (§0). |
| Recombined founder FASTAs | `cross/write_fastas.py` | Concatenates parent assembly sequence per mosaic. |
| PS4G parsing | `ps4g_io/ps4g.py` | `load_ps4g_file`, `extract_metadata` (gamete↔index map). |
| PS4G → training matrix | `cross/build_training_data.py`, `cross/ps4g_to_matrix.py` | Answer-key build, position collapse, all-positions fill, optional positional-embed column. |
| Diploid CRF model | `crf/train_crf.py` | `FounderPathEncoder` + pair-state `NeuralCRF` + `diploid_emissions`/`diploid_pair_index`. |
| Lightning training loop | `crf/train_crf.py` (`GRITSCRFModel`, `main`) | AdamW + ReduceLROnPlateau, gate reg, TB logger. |
| Smoke/pressure test | `crf/pressure_test.py` | Env/data/forward/backward/throughput; writes `results/pressure_test.txt`. |
| HMM baseline | `hmm/hmm_impute.py`, `hmm/viterbi.py` | Exists; **not yet wrapped** to score on the shared held-out set (see §2.2). |
| GRITS production model | `supervised/` (seq2seq + `modernBERT/` + `bimamba/`) | The ~43M reference; **not yet wrapped** for re-scoring (see §2.2). |

### 2.2 Still open (the real `TODO(wire)` list now)

- `TODO(wire)`: **Baseline scoring wrappers** — run the existing HMM and the GRITS
  production model on **one shared held-out sim set** and emit the §4.3 metrics.
  This is `baselines.py` and does not exist yet.
- `TODO(wire)`: **Compute budget** — record GPU model / VRAM actually available so
  `d`/`L`/window and Transformer-vs-Mamba2 (E6) are sized against real limits.
- `TODO(decide)`: **Held-out split policy** — the simulator can produce arbitrary
  panels; define and freeze the train / val / held-out (and unseen-panel) split so
  every experiment scores on identical data.

### 2.3 Known code debt (tracked, NOT in scope for this doc pass)

Per the current session decision, the plan does not modify code. These are recorded
so experiments account for them and a later cleanup pass can target them:
- ~~**No haploid code path yet.**~~ **RESOLVED (2026-06-23):** `train_haploid.py`
  provides the single-hap dataset/label path (`PreWindowedHaploidDataset`) and the
  `K`-state `nsw`, feeding the same shared `FounderPathEncoder` + `NeuralCRF`.
  `train_crf.py` (diploid) still collapses labels to pair indices — unchanged.
- `forward()` is invoked twice per `training_step` in `train_crf.py`.
- Debug `print()` calls fire inside `forward()` every pass.
- Hardcoded data paths + inline `config` dict in `train_crf.main()`; no YAML yet.
  (`train_haploid.py` **does** use argparse + `--workdir`; the diploid entry point
  still needs it.)
- Import convention is split repo-wide: `crf/` uses `from python.<module>`,
  `cross/` and several `supervised/` files use `from src.python.<module>`. Pick one
  before `baselines.py` imports across both trees.

---

## 3. Repo layout (actual + target)

```
src/python/
  crf/
    train_crf.py        # BUILT: FounderPathEncoder, NeuralCRF, GRITSCRFModel (diploid)
    pressure_test.py    # BUILT: smoke/throughput harness
    metrics.py          # TODO: founder_acc, switch_error, breakpoint_loc, snp_acc, profiler
    baselines.py        # TODO: run/score HMM + GRITS on shared held-out
    configs/            # TODO: one yaml per experiment (E1..E7)
  cross/                # BUILT: simulator + PS4G→matrix pipeline
  hmm/                  # BUILT: Li–Stephens baseline (needs scoring wrapper)
  supervised/, modernBERT/, bimamba/   # BUILT: ~43M GRITS reference stack
tests/python/crf/       # TODO: overfit, viterbi-vs-brute-force, gradient, shape tests
docs/RESULTS.md         # TODO: the running leaderboard (template in §7)
```

---

## 4. Interface contracts

### 4.1 Data sample (as produced by `cross/` + consumed by `LabeledDatasetDiploid`)

A training matrix on disk is `np.int8` of shape `[T, K + L]`:
- columns `0 : K` — read-count support of nominated founder *k* at site *t*
  (`0` = miss, clipped to `127`), where `K = num_parents` (default 24).
- optional positional-embed column (when built with `--positional-embed`), inserted
  **after features, before labels**.
- final label column(s): **2** for diploid (`[T, K+2]`), **1** for haploid
  (`[T, K+1]`). `-1` denotes an unlabeled/unknown bin and is mapped to the
  **unknown state index `K`** by the dataset.

`LabeledDatasetDiploid` windows this (`window_size=512`, `step_size=128`) and converts
the two diploid label columns into a single **unordered pair-state index** via
`create_diploid_pairs(num_parents)` → `P = (K+1)(K+2)/2` states (the `+1` is the
unknown founder). Imperfect nomination and "truth = best available founder" are
handled upstream in the answer-key build.

For the **haploid milestone (E1)** the single `[T, K+1]` matrix is used directly:
tags are the lone label column in `[0, K)` (with `-1`→`K`), no pair conversion. The
needed single-hap dataset path is the §2.3 addition; the feature columns and windowing
are identical to the diploid case.

### 4.2 Model (in `crf/train_crf.py`, keep this public interface stable)

```python
enc = FounderPathEncoder(d_model, n_heads, n_layers, ext_dim=0)
emis_f, g, c = enc(X, founder_mask, dbp=None, ext_emb=None)   # emis_f[B,T,K], g[B,T], c[B,T]

# Haploid milestone (E1): CRF directly over K founder states.
nsw_h = (1.0 - torch.eye(K))                                  # nsw[i,j] = (i!=j)
tr_h  = crf._trans(c, nsw_h)                                  # [B,T,K,K]
loss  = crf.nll(emis_f, tr_h, tags_haploid)                   # tags_haploid in [0,K)
path  = crf.viterbi(emis_f, tr_h)

# Diploid path (later experiments): same encoder, same crf, pair-state layer.
emis_p = diploid_emissions(emis_f, pair_idx)                  # [B,T,P]
tr_p   = crf._trans(c, nsw)                                   # nsw = min-over-orderings
loss_p = crf.nll(emis_p, tr_p, tags_pair)
```

`NeuralCRF` is **state-count-agnostic** — `_trans/partition/score/viterbi/nll` take
any `emis [B,T,S]` and `nsw [S,S]`, so haploid (`S=K`) and diploid (`S=P`) reuse the
identical implementation. `g` is the reliability gate (kept near 1 by `gate_reg`);
`c` is the per-site recomb cost feeding the transition. `ext_dim` is the hook for
the relatedness / external-embedding work described in `CLAUDE.md`.

### 4.3 Metrics (`metrics.py`, TODO)
- `founder_accuracy(pred, tags, mask)` — per-site, panel-masked, on the **pair state**.
- `switch_error_rate(pred, tags)` — per informative interval.
- `breakpoint_localization(pred, tags, tol_bp)` — precision/recall within `tol_bp`;
  also report **false breakpoints inside `meta.sv_blocks`** (key for E3).
- `snp_accuracy_from_path(pred, panel)` — reconstruct SNP dosages from the path,
  MAF-binned, for apples-to-apples vs SNP imputers.
- `param_count(model)` + a `(window_len, K, peak_mem, throughput)` profiler.

---

## 5. Experiment series (diploid-first)

Run in order. Each experiment = one config + one script + one row in `RESULTS.md`,
and must include the **ablation** named in its acceptance criteria. **E1 trains and
validates the haploid model first**; E2–E4 are run haploid (cheaper, fewer states);
E4b promotes to the diploid pair-state path on the same architecture; E5–E8 build on
whichever state layer each targets. The encoder and `NeuralCRF` are shared across all
of them (§4.2). **Numbering (2026-06-25):** E5 = relatedness/genome-wide conditioning
(in progress), E6 = SNP-level accuracy, E7 = mixed-inbreeding tuning, E8 = inference
speed; the former standalone Mamba2 milestone is folded into E8.

### E0 — Harness & baselines (no modeling)
- **Objective:** make every later number interpretable.
- **Implement:** `metrics.py` (§4.3); `baselines.py` wrappers to run/score the
  existing **HMM** and **GRITS** on one shared held-out sim set; freeze the split
  (§2.2 `TODO(decide)`); seed `RESULTS.md`; add `tests/` (golden-path overfit that
  drives CRF NLL→~0 on a single batch, Viterbi-vs-brute-force on a tiny pair space,
  a gradient/finite-difference check on the NLL, shape tests).
- **Acceptance:** identical held-out set scored by both baselines on all §4.3
  metrics; GRITS param count reproduced (~43M); overfit + Viterbi tests pass.
- **Decision informed:** baseline bar + measurement validity. **This is the gate.**

### E1 — Haploid training milestone + structured-layer ablation (the central test)
- **Status (2026-06-23):** *in progress.* Encoder trains end-to-end on the synthetic
  allele-sharing sim (val acc 0.99, §0). **Still open before E1 closes:** HMM baseline
  + `metrics.py` (E0 gate), training on the `cross/` data source, the no-transition
  ablation arm, and the E4b diploid-load check.
- **Objective:** get the system trained end-to-end on **haploid** data, and answer
  the central question — does the neural-CRF prior match/beat the HMM at small param
  count? This is the first real training run.
- **Implement:** add the single-hap dataset/label path and the `K`-state `nsw`
  (§2.3, §4.2); train the shared `FounderPathEncoder + NeuralCRF` over `K` founder
  states on **un-split** chromosomes (one recombination regime). Ablation arm:
  **emission-only / no transition** (`c→∞` so switches are free → per-site argmax) to
  isolate the CRF. **Architecture check:** the same checkpoint's encoder must drop
  into the diploid pair-state path with zero structural change (verified in E4b).
- **Run:** `configs/e1.yaml` at `d=256, L=6` (~5M params; see §6).
- **Acceptance:** (a) haploid founder accuracy ≥ HMM on held-out; (b) params ≤ ~6M
  (≈8× under GRITS); (c) full CRF beats the no-transition ablation by a clear margin;
  (d) the trained encoder loads unchanged into the diploid wrapper.
- **Decision informed:** is "neuralize the HMM" the right direction at all?

### E2 — Recombination-rate heterogeneity
- **Status (2026-06-23):** *next up* — the "Step 2" in §0. On the synthetic sim, `c_t`
  collapsed to a constant (`c≈5`) because recombination there is spatially uniform;
  E2 adds a ~100× variable rate map (in `simulate_alleles.py` and/or `cross/`) so
  `c_t` has signal to track.
- **Objective:** replace hand-split regions with learned `c_t`.
- **Implement:** train across simulated rate regimes spanning ~2 orders of magnitude
  (vary `min_spacing`/`max_spacing` in `pick_crossovers`); feed `dbp` into the
  recomb head. Ablation arm: **constant transition** (learned scalar, no per-site `c_t`).
- **Run:** sweep crossover frequency; plot accuracy-vs-crossover-frequency.
- **Acceptance:** learned `c_t` flattens that curve vs the constant arm; matches an
  HMM given the *true* per-site map without being told it.
- **Decision informed:** is learned `c_t` enough, or is an explicit map covariate needed?

### E3 — SV / bad-nomination robustness + reliability gate
- **Objective:** stop spurious breakpoints inside correlated-bad-data blocks.
- **Implement:** inject SV/indel and bad-nomination blocks in sim (record spans in
  `meta.sv_blocks`); keep `tags` constant through them. Ablation arms: **gate off**
  (`g≡1`) and **gate without budget reg** (test for collapse).
- **Run:** `configs/e3.yaml`; report false-breakpoint rate *inside* injected blocks.
- **Acceptance:** gate reduces in-block false breakpoints vs gate-off **without**
  raising real-breakpoint error; gate does not collapse to ~0; no identifiability
  blow-up between low `g` and high `c` (inspect their correlation).
- **Decision informed:** is the gate+`c_t` mechanism well-posed?

### E4b — Promote to diploid pair-state
- **Objective:** carry the validated haploid encoder into the joint diploid decode and
  confirm the pair-state model beats independent per-hap decode — cashing in the
  "diploid-capable from the start" invariant.
- **Implement:** load the E1 encoder unchanged into the pair-state wrapper
  (`diploid_emissions` + `P`-state `nsw`); train/fine-tune over pair states. Ablation
  arm: **two independent haplotype decodes + post-hoc pairing** (i.e. the E1 model run
  twice). Verify the transition uses the matrix-free stay/switch recursion (do **not**
  materialize `[B,T,P,P]`).
- **Run:** K=24; profile decode cost `O(T·P²)` and memory.
- **Acceptance:** lower switch-error than the independent-decode arm; tractable at
  K=24 and target chromosome length; **zero added parameters** vs the E1 encoder
  (verify the diploid path introduced no new weights).
- **Decision informed:** does joint pair-state earn its cost over per-hap+coupling;
  tractability ceiling on K.

### E5 — Relatedness / genome-wide founder conditioning
- **Status (2026-06-25): in progress — the `crf-relatedness` branch core** (this
  subsumes the former "E7 global conditioning" and the relatedness-matrix arm of the
  former "E5 nomination transfer"; renumbered to match the executed series).
- **Objective:** use a per-individual, genome-wide founder relatedness/presence
  signal to disambiguate locally IBD-ambiguous calls — i.e. **break the read-only
  IBD ceiling** characterized in RESULTS "E-IBD".
- **Implement:** sim groups windows into **individuals** over a 2–24 founder subset
  (`--windows-per-individual`); a per-individual, per-founder **affinity vector**,
  estimated from reads only (no labels), conditions the encoder via `ext_emb`
  (zero-init projection; bounded features to avoid bf16 partition overflow). Arms:
  relatedness **on/off**; later, transfer to **unseen founder compositions**
  (index-agnostic claim) and an optional frozen external assembly embedding
  (PlantCAD2) via `ext_dim`.
- **Run:** train on the grouped sim (`sim_e5_th6`); eval with `eval_e5.py` against
  the read-only ceiling AND the relatedness ceiling (`relatedness_ceiling.py`).
- **Acceptance:** the relatedness arm **exceeds the read-only IBD ceiling** in the
  hard bands (headroom measured at +5.2pt overall / +7.9pt in 6+bp) without hurting
  easy windows; the off-arm sits at the ceiling.
- **Decision informed:** does genome-wide conditioning recover IBD-confusable calls,
  and by how much of the measured headroom.
- **Files:** `simulate_alleles.py` (grouping), `train_haploid.py`
  (`IndividualRelatednessDataset`, `make_individual_splits`), `eval_e5.py`,
  `relatedness_ceiling.py`.

### E6 — SNP-level imputation accuracy (benchmark-comparable metric)
- **Objective:** report per-SNP **genotype concordance** and **dosage r²** at masked
  sites so GRITS is directly comparable to standard imputers (Beagle / minimac /
  IMPUTE5), and demonstrate that the IBD *path* ceiling does **not** cap SNP
  accuracy — path errors land on IBD-confusable founders that share alleles over
  the block, so the SNP call is usually still correct. We model/train on founder
  haplotypes; we *report* SNPs by composing the decoded path with a founder panel.
- **Implement (sim, `simulate_alleles.py`):** in coalescent mode, persist an
  eval-only **founder × SNP allele panel** + the sample's true genotype as a
  companion (`<out>.snp.npy`, test-split only, bit-packable) from the per-SNP
  mini-haplotype alleles already generated (`lin_alleles` / `G` in
  `_coalescent_feats`). Add `--mask-frac`: split SNPs into **typed** (drive the
  match features as today) vs **masked** (hidden from input, scored at eval) — the
  real imputation setting, mirroring how other tools mask a typed-array subset and
  impute the rest. Granularity is **per biallelic SNP** (not per mini-haplotype
  read) for literature comparability.
- **Eval (`eval_snp.py`):** `predicted_allele(s) = panel[decoded_founder(s), s]`;
  diploid composes H1+H2 into 0/1/2 **dosage**. Report concordance + dosage r²
  **stratified by MAF**, alongside founder-path accuracy on the same windows so the
  path-vs-SNP gap (the IBD-invisible-at-SNP effect) is explicit.
- **Acceptance:** SNP accuracy ≫ path accuracy with the gap explained by IBD
  allele-sharing; competitive concordance / r² vs a published baseline on a matched
  MAF spectrum; metric is per-SNP.
- **Decision informed:** is GRITS's SNP-level imputation competitive, and how much
  of the path-ceiling gap is invisible at the SNP level.

### E7 — Mixed-inbreeding panel tuning (variable F across individuals)
- **Objective:** tune one model to serve a realistic panel that **mixes inbreeding
  levels** — fully inbred lines (F≈1, identical gametes → haploid-like), outbred
  (F≈0, interleaved single-gamete reads needing phasing), and everything between —
  rather than the single-F sims used so far. This is the main pre-productionization
  modeling task.
- **Implement:** draw a **per-individual** inbreeding coefficient F from a
  distribution (e.g. a spike at F=1 for inbred lines plus a spread over [0,1])
  instead of the single `--inbreeding` scalar; train the diploid pair-state model
  (E4b) on the mixture, reusing the E5 individual grouping. Ablation arms:
  single-F-trained model evaluated on the mixture; with/without the E5 relatedness
  signal under varying F.
- **Run:** `configs/e7.yaml`; sweep the F distribution; report pair/hap accuracy and
  SNP r² **stratified by F**.
- **Acceptance:** a **single** model handles the full F range with no per-F
  recalibration — low-F individuals keep diploid phasing accuracy while inbred
  individuals stay at the haploid ceiling; beats the single-F-trained baseline on
  the mixed panel.
- **Decision informed:** can one GRITS model serve mixed-inbreeding panels (the
  real-data setting), or is inbreeding-stratified training required.

### E8 — Inference speed / make-it-fast (final productionization)
- **Objective:** minimize end-to-end decode **latency and peak memory** at target
  chromosome length, at no meaningful accuracy cost — the last task before release.
- **Implement (decide each empirically; keep the §4.2 CRF interface unchanged):**
  profile the Transformer encoder + Viterbi decode first, then apply the levers that
  the profile justifies — (a) **long-context encoder swap to Mamba2** (`bimamba/`
  reference) **only if** the Transformer is the length/memory bottleneck; (b)
  decoder/kernel optimization (fused stay/switch recursion, batched Viterbi);
  (c) reduced-precision / quantized inference; (d) window-stitching for a
  single-pass full-chromosome decode. Assert **param count invariant to T and K** in
  `tests/`.
- **Run:** sweep window / chromosome length; report throughput (sites/s) and peak
  memory vs accuracy.
- **Acceptance:** target chromosome decodes within the §2.2 GPU budget with
  accuracy within a pre-registered margin of the unoptimized model.
- **Decision informed:** final encoder choice and max single-pass window — **and
  whether Mamba2 is needed at all** (adopt only if it wins the profile; the former
  standalone "Mamba2" milestone is demoted to one candidate lever here).

### E9 — Report writeup + figures
- **Objective:** a self-contained `docs/REPORT.md` that tells the project story
  with publication-style figures, suitable for collaborators / a methods writeup.
- **Implement:** a reproducible `crf/figures.py` that regenerates every figure from
  the `results/` tables and the sims, writing PNGs to `<workdir>/results/figures/`.
  Required figures: (1) **simulator schematic** — founders → GEM/coalescent IBD
  lineages → recombination-mosaic paths → per-individual founder subset (E5) →
  single-gamete reads → match-feature matrix; the inbreeding mix (E7) and the
  het-from-adjacent-read-correlation signal. (2) **model architecture** — cell
  embeddings → founder attention pool → Transformer → per-site emissions + gate +
  transition cost c_t → NeuralCRF Viterbi; the diploid pair-state and the E5
  affinity / E7 het hooks. (3) **key results**: CRF-vs-HMM (E1), IBD ceiling +
  θ-sweep (E-IBD), relatedness cutoff vs ceiling (E5), SNP≫path (E6), accuracy-by-F
  for the three penalty arms (E7), speed/memory scaling (E8).
- **Acceptance:** `REPORT.md` renders with all figures; every figure is
  regenerable by one `figures.py` run from committed results; numbers match
  RESULTS.md.
- **Decision informed:** is the story complete and communicable end-to-end.

### E10 — Per-read position: disambiguate heterozygosity from recombination
- **Motivation:** in the diploid eval the model recovers ~1.3 of 2 founders per het
  site (`hap_acc` ≫ `pair_acc`) — *not* a phasing problem (pair-states are unordered),
  but the **single-gamete coverage limit**: one read per site means the second homolog's
  founder is unobserved and must be inferred, and the inference confounds *het* with
  *recombination*. The current sim emits one read per evenly-spaced integer site, so the
  signal that separates them never appears.
- **Key idea (zrm22):** give each read a **relative position float ∈ [0,1]**. Two
  conflicting reads **close/identical in position ⇒ heterozygous** (two homologs at one
  locus — both founders directly observed); **far apart ⇒ recombination** (one homolog
  switched). At co-located conflicts the emission should favor the het pair-state {A,B}.
- **Implement:**
  1. *Simulator* — per-read layout: reads drawn from BOTH gametes at Poisson-spaced
     positions, each `(position, founder-match vector)`; read density a knob (doubles as
     a coverage sweep). Keep eval-only H1/H2 paths + IBD. (The per-read, not
     collapsed-[T,K], layout already flagged as a direction.)
  2. *Encoder* — embed the position float (Fourier features / small MLP) and add it to
     each read's cell embedding, so attention pooling over co-located reads can drive
     the het pair-state.
- **Acceptance:** few-founder outbred `pair_acc` (today's OOD baseline 0.37–0.40) rises
  toward `hap_acc` as read density increases; the model separates het from recombination
  (e.g. lower breakpoint false-positives at het runs).
- **Decision informed:** does observing the second homolog (vs inferring it) close the
  pair-vs-hap gap — i.e. is the diploid limit coverage-bound, as hypothesised.

---

## 6. Default config & parameter budget

Trainable params ≈ `(5 + 12·L)·d² + (10 + 13·L)·d + 6` (exact for the Transformer
encoder). Reference points: `d=96,L=2 → 271K`; `d=128,L=4 → 876K`;
**`d=256,L=6 → 5.07M` (default for E1+)**; `d=384,L=12 → 22M`. **Independent of T
(sites) and K (founders); diploid pair-state adds 0 params.** Start at 5M; only
scale `d`/`L` if E1 underfits. (Note: `train_crf.main()` currently ships a smaller
`d=64,L=2` debug config — E1 must override to the 5M default via its config.)

---

## 7. RESULTS.md leaderboard template (`docs/RESULTS.md`, TODO)

| Exp | Model / arm | Params | Founder acc | Switch err | Breakpt P/R | In-SV false bp | SNP acc (MAF) | Mem / it/s |
|-----|-------------|-------:|------------:|-----------:|-------------|----------------|---------------|------------|
| E0  | PHG-HMM     |   n/a  |             |            |             |                |               |            |
| E0  | GRITS       |  ~43M  |             |            |             |                |               |            |
| E1  | CRF (full)  |  ~5M   |             |            |             |                |               |            |
| E1  | CRF (no-transition) | ~5M |        |            |             |                |               |            |

---

## 8. Engineering conventions
- Config-driven (one yaml per experiment under `crf/configs/`); log seed, git SHA, config.
- Fixed seeds; report mean ± sd over ≥3 seeds for any headline comparison vs HMM.
- `tests/` must include: shape tests, a gradient/finite-difference check on the CRF NLL,
  a Viterbi-vs-brute-force agreement test on a tiny pair-state space, and the golden-path
  overfit test from E0.
- Every experiment appends to `RESULTS.md`; never overwrite prior rows.
- Keep the model module's public interface (§4.2) stable across experiments.
- Write all data/checkpoints/logs/results under an explicit `--workdir` (see `CLAUDE.md`).

## 9. Guardrails / non-goals
- Do **not** predict SNPs as the training target; the target is the path. (SNP accuracy
  is a *reconstructed* downstream metric only.)
- Do **not** add a per-founder-index parameter anywhere — it breaks variable-panel transfer.
- Do **not** materialize `[B,T,P,P]` transitions in the diploid decode; use the
  matrix-free stay/switch recursion already in `NeuralCRF._trans`.
- Do **not** tune against the test split or the real-data eval; reserve those.

## 10. Settled vs. open decisions
- **Settled:** first training milestone is **haploid**, on a **diploid-capable**
  architecture (shared encoder + state-agnostic CRF; promote to pair-state in E4b
  with no encoder rework); path restricted to nominated top-K; no off-panel state;
  truth = best available founder; nomination simulated inside the pipeline;
  `c_t`/`g_t` learned end-to-end (never directly supervised); the `cross/` pipeline
  is the data source of record.
- **Open (resolve via experiments):** Transformer vs Mamba2 (E6); whether the joint
  pair-state decode beats per-hap+coupling (E4); whether global conditioning earns
  its place (E7); whether an explicit recombination-map covariate is needed beyond
  `c_t` (E2); value of an external founder/relatedness embedding (E5).
```