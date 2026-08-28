# Tier 2 evaluation: constant-pair/constant-inbred individuals

Companion to `results/tier1_evaluation.md` and
`/home/zrm22/.claude/plans/wondrous-discovering-octopus.md`. Full raw table:
`results/tier2_evaluation_raw.md`.

**Checkpoints compared** (all `--founder-affinity --learned-het
--time-local-emis`, same stability recipe):
- `diploid-affinity-sim512-h3` — deployed baseline, fixed `homo_penalty`.
- `breedpop-sparse-affinity-learnedhet` — Tier 1, sparse crossovers only.
- `breedpop-sparse-constant-affinity-learnedhet` — Tier 2, Tier 1's data plus
  30% of individuals given a genuinely constant founder identity (0
  breakpoints, exact het=0 or het=1), via a new, verified
  `--constant-pair-frac`/`--constant-inbred-frac` addition to
  `simulate_alleles.py` (branch `tier2-constant-pair-individuals` off
  `origin/develop`). Sim val_pair_acc **0.9215** — not directly comparable
  to Tier 1's 0.7868, since ~30% of this validation split is now trivially
  easy (0-breakpoint constant individuals), inflating the aggregate.

**The simulator change itself is verified correct.** Smoke test (20
individuals) and a 30-individual spot-check on the full 100k-window run
both confirm the exact target invariant: every constant individual decodes
to exactly 0 switches; constant-inbred individuals hit het_frac=0.0000
exactly (k=1, never reachable before); constant-het individuals hit
het_frac=1.0000 exactly (k=2, previously capped at 0.5 — see Tier 1's
writeup for why).

## Headline result: **the specific hypothesis was NOT confirmed**

The prediction going into Tier 2 (from Tier 1's `homo_mass_pen` finding)
was that adding genuinely 0-breakpoint, fully-het/fully-homo individuals
would recalibrate the learned-het head and recover the hybrid/inbred
regression. It didn't, materially:

| row | metric | baseline | Tier 1 | Tier 2 | Tier2 − Tier1 |
|---|---|---|---|---|---|
| Hybrid | pair_acc | 0.828 | 0.618 | 0.605 | **−0.013** |
| Hybrid | homo_mass_pen (mean) | 0.003 | 0.264 | 0.261 | **−0.003** (essentially flat) |
| Inbred Oh43 | pair_acc | 1.000 | 0.840 | 0.840 | **+0.000** (identical) |
| Inbred Il14H | pair_acc | 1.000 | 0.900 | 0.900 | **+0.000** (identical) |
| RIL | pair_acc | 0.546 | 0.683 | 0.697 | +0.014 (modest further gain) |
| RIL homo/het gap | | −0.575 | +0.083 | +0.058 | slightly more balanced |

Inbred rows are bit-for-bit identical between Tier 1 and Tier 2 at this
sample size — the 15% of training individuals that are now genuinely
constant-inbred had **no measurable effect** on real inbred decoding.
Hybrid's `homo_mass_pen` — the direct mechanism metric this tier was built
to move — barely changed (0.264→0.261). RIL continued to improve modestly,
consistent with Tier 1's direction, but that's a continuation of the
general breeding-pop-mixture effect already established, not something
newly attributable to the constant individuals specifically.

**A real, secondary effect the mean hides**: `homo_mass_pen`'s
distribution shifted even though its mean didn't. On the real hybrid,
Tier 2 has a *lower* median (0.018 vs 0.032) and a *larger* fraction of
sites confidently correct (63.9% below 0.1, vs 61.0%) — but also a
slightly *larger* fraction confidently wrong (25.4% above 0.5, vs 24.6%).
Tier 2 sharpened the model's decision boundary without improving its
average calibration: it more strongly recognizes windows that resemble the
constant training pattern, and is no better (if anything slightly worse)
on windows that don't.

## What this implies

The working explanation from Tier 1 — "the learned-het head hasn't seen a
genuinely 100%-heterozygous individual, so it's under-confident there" —
was too simple. Adding that exact regime to training didn't fix real
hybrid decoding. Two more likely explanations, neither yet tested:

1. **Sim-to-real feature-distribution gap, not a training-regime gap.**
   The learned-het head reads local read-alternation *patterns* from the
   encoder's hidden state. A real hybrid's alternation pattern (from actual
   ropebwt3-aligned reads, real sequencing noise, real homology structure)
   may simply look different at the feature level from even a "correctly
   structured" simulated constant individual (same generative mechanism —
   `_coalescent_feats` — as everything else in this sim), independent of
   whether the *label* structure (0 breakpoints, het=1) matches. This
   would mean the fix isn't about which *label regimes* exist in training,
   but about how well the *feature generator itself* mimics real
   ropebwt3/refmap output — a different, harder problem than Tier 2 addressed.
2. **Founder-affinity interaction.** `ext_emb`/`ext_bias` is a genome-wide,
   per-individual signal computed once and added to every site. A
   genuinely 2-founder-only constant individual produces a very different
   (sharper, more binary) affinity vector shape than a real hybrid's
   ropebwt3-derived affinity vector does, even though both nominally
   represent "2 true founders." Not directly tested this round.

Both point toward the same practical conclusion: **further training-data
regime engineering (more Tier-2-style additions) is unlikely to be the
highest-leverage next step** for the hybrid/inbred regression specifically.
RIL continues to respond to this general approach; hybrid/inbred do not.

## Practical recommendation

Given the deployed checkpoint (fixed `homo_penalty`, externally-supplied
`homo_scale`) is *already* correct for hybrid and inbred (`homo_scale`
is *told*, not inferred) and only genuinely fails on RIL's within-genome
zygosity mixing, and given neither Tier 1 nor Tier 2's learned-het
checkpoint recovers hybrid/inbred while both improve RIL — the two
regimes may simply call for different inference strategies rather than
one model: keep the deployed checkpoint (or its `homo_scale` mechanism)
for known-inbred/known-hybrid samples, and reach for a learned-het
checkpoint specifically for RIL-like (unknown or mixed-zygosity) samples.
This wasn't tested or built this round — flagged as the more promising
next avenue than a further Tier 3 data-regime change.

## Files

- Branch: `tier2-constant-pair-individuals` (off `origin/develop`),
  worktree `/local/workdir/zrm22/HackathonJun2026/grits-tier2-worktree`,
  commit `a5a0587`.
- Data: `data/training/sim_breedpop_sparse_constant.npy`
  (+`.ind/.finb/.cls.npy`, class id 3 = constant individuals).
- Checkpoint: `checkpoints/breedpop-sparse-constant-affinity-learnedhet/`
  (best: `d-epoch=01-val_pair_acc=0.9215.ckpt`).
- Extended `scripts/compare_checkpoints.py` to a multi-tag (`--tags`)
  3-way comparison, including a new `homo_mass_pen (mean)` row per metric
  table — the direct mechanism check, now systematic rather than ad hoc.
- Raw dumps: `results/model_internals/breedpop-sparse-constant-affinity-learnedhet/*.npz`.
