# Tier 1 evaluation: sparse-crossover breeding-pop retrain vs deployed checkpoint

Companion to `results/tier0_findings.md` and
`/home/zrm22/.claude/plans/wondrous-discovering-octopus.md`.

**Baseline**: `checkpoints/diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt`
(fixed `homo_penalty=3.0 * homo_scale`, no learned-het, trained on
`--min-founders 2 --max-founders 24 --min-crossovers 1 --max-crossovers 4`
independent-window data).

**New**: `checkpoints/breedpop-sparse-affinity-learnedhet/d-epoch=02-val_pair_acc=0.7868.ckpt`
(`--founder-affinity --learned-het --time-local-emis`, trained on
`sim_breedpop_sparse.npy` — `--breeding-pop --min-crossovers 0
--max-crossovers 1 --het-by-class 0.5,0.25,0.9 --class-inbred-frac 0.5`).
Sim-side val_pair_acc **0.7868**, above even E11-affinity's historical best
(0.7587, `docs/RESULTS.md:1062-1070`) and far above the deployed checkpoint's
0.6179. Trained cleanly, 3 epochs, best-val checkpointing captured epoch 2.

Real-data evaluation run twice (24 windows, then 100 windows per row) to
confirm the pattern isn't a small-sample artifact — it isn't; n=100
sharpens the same story.

## Headline numbers (n=100 windows/row; `results/tier1_evaluation_n100.md` has the full table)

| row | metric | baseline | new | delta |
|---|---|---|---|---|
| IDX-INBRED Oh43 | pair_acc | 1.000 | 0.840 | **−0.160** |
| IDX-INBRED Il14H | pair_acc | 1.000 | 0.900 | **−0.100** |
| IDX-HYB Oh43xIl14H | pair_acc | 0.828 | 0.618 | **−0.210** |
| IDX-HYB Oh43xIl14H | decoded switches/window | 1.58 | 0.51 | **−1.07** |
| IDX-RIL Oh43xIl14H | pair_acc | 0.546 | 0.683 | **+0.137** |
| IDX-RIL Oh43xIl14H | decoded switches/window | 2.48 | 0.22 | **−2.26** |
| IDX-RIL homo-truth windows | pair_acc | 0.218 | 0.730 | **+0.512** |
| IDX-RIL het-truth windows | pair_acc | 0.793 | 0.647 | **−0.146** |

## Three real, independently-confirmed effects — not all in the same direction

**1. H2 (path-structure mismatch) — decisively fixed.** Decoded switches per
window dropped 3–11× everywhere real or matched-sim data was tested (hybrid
1.58→0.51, RIL 2.48→0.22, sim k=2 4.63→1.00). Training on sparse crossovers
(mean 0.5/window vs the old ~5/window) taught the CRF transition structure
to stop over-switching. Unambiguous, large, consistent at both sample sizes.

**2. H3-per-locus (RIL's fixed-homo_scale problem) — substantially fixed,
with a new, informative wrinkle.** RIL pair_acc improved +13.7pp overall.
The homo/het truth-conditioned split (the direct mechanism test from Tier
0) moved from a −57.5pp gap (baseline: homozygous-truth windows score far
worse, 0.218 vs 0.793) to a **+8.3pp gap in the OPPOSITE direction** (new:
homo-truth 0.730 vs het-truth 0.647) — the per-locus learned prior didn't
just narrow the gap, it overshot past parity to a mild bias the other way.
Net effect is strongly positive for RIL (both classes improved in absolute
terms; homo-truth alone gained +51pp), but the direction-flip is worth
flagging for anyone tuning further: the learned-het head is not yet
perfectly calibrated at either true extreme.

**3. Hybrid AND inbred regressed — root cause confirmed directly, not
inferred.** Real hybrid pair_acc fell 0.828→0.618 (−21pp); both inbred rows
fell 8–16pp too. Direct mechanism check (`homo_mass_pen`, the model's own
posterior mass on any homozygous state): on the real, genuinely
100%-heterozygous hybrid, the baseline checkpoint sits at **1.2% mean**
homozygous mass (correctly, confidently heterozygous everywhere, because it
was *told* `homo_scale=1.0` — an externally-supplied, ground-truth-informed
signal). The new checkpoint's learned-het head, which must infer
zygosity from reads alone, sits at **20.2% mean** homozygous mass on the
same row (19.6% of sites actually lean homozygous). This is the same
structural gap flagged *before* training even started: no individual in
`sim_breedpop_sparse.npy` (or reachable via any existing flag combination,
per the plan's Tier 2 discussion) is genuinely 100% heterozygous with 0
breakpoints — the k=2 het ceiling is structurally 0.5. The learned-het head
has simply never seen the extreme case a real F1 hybrid actually is, and
under-fires there. Inbred's regression is a smaller instance of the same
gap from the other side (still not literal k=1 in training, though
`--class-inbred-frac` gets much closer than before).

## Net reading

Tier 1 is a real, partial win, not a clean one: it fixes H2 conclusively,
substantially improves RIL (the dataset closest to a "typical" real
individual, and the one degraded worst going in), but costs hybrid and
inbred accuracy by removing the externally-supplied `homo_scale` signal in
favor of a read-inferred one that isn't yet calibrated at the extremes
training still can't produce. **This sharpens, rather than replaces, the
case for Tier 2** (the constant-pair/true-k=1/true-het=1 simulator addition
already scoped in the plan) — the hybrid/inbred regression is a direct,
measured consequence of the exact gap Tier 2 targets, not a new problem.

**Not promoted to production.** `heldout_assembly_eval.py`/
`simval_eval_one.py`/`nam_diploid.py` still point at the deployed
`diploid-affinity-sim512-h3` checkpoint — that hardcoded path is unchanged.
Promoting `breedpop-sparse-affinity-learnedhet` as-is would trade a
non-trivial hybrid/inbred regression for a RIL/H2 win; not a clear
improvement until Tier 2 closes the remaining gap.

## Files

- Data: `data/training/sim_breedpop_sparse.npy` (+`.ind/.finb/.cls.npy`)
- Checkpoint: `checkpoints/breedpop-sparse-affinity-learnedhet/`
- Scripts: `scripts/homo_penalty_sweep.py`, `scripts/whole_run_decode_check.py`,
  `scripts/dump_model_internals.py` (generalized: `--ckpt`, RIL truth wired
  up via `_window_ril_truth`, `--learned-het`-aware penalty reconstruction),
  `scripts/compare_checkpoints.py`
- Raw dumps: `results/model_internals/{diploid-affinity-sim512-h3,
  breedpop-sparse-affinity-learnedhet}/*.npz` — n=100 windows/row (the n=24
  first-pass dumps were overwritten in place by the n=100 re-run, same
  filenames; the n=24 numbers quoted above and in `results/tier0_findings.md`
  came from console output captured before the overwrite, not from re-reading
  these files)
- `results/tier1_evaluation_n100.md` — full per-metric table
