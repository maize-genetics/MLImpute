# Why chain's B73 hit_ratio isn't a similarity measure: a source-level root cause

Follow-on to [[refbias_eval]] / `results/refbias.md`. The user reviewed the
original eval and flagged that `chain`'s B73 `hit_ratio` (65-75% of placed
reads) looked implausibly high against the expectation that ~40% PAV
divergence between any two maize lines should cap B73's apparent share at
roughly 60%. Investigated read-only, zero new alignment runs — everything
below is derived from the 120 `.ps4g` files already on disk from the
original eval, two `chain_nolift` baselines cached from an earlier session,
and a direct read of `ropebwt3-phg`'s `search.c`
(`.claude/worktrees/pav-ps4g-insertion-rows`).

## Question 1: how many reads are PAV hits, and what fraction?

**27-35% of chain's placed reads are PAV-recovered**, excluding the
degenerate B73-own-reads case (where PAV is trivially ~0.3-0.4%, since B73's
own reads obviously match B73 almost everywhere):

| arm | mean PAV% of placed | range |
|---|---:|---|
| `chain_ordinary_5000` (looser gate) | 33.8% | 28.9-39.6% |
| `chain_ordinary_256` (tighter gate) | 29.4% | 24.6-35.3% |

By germplasm class (`chain_ordinary_256`): indexed 31.9%, mixed 29.7%,
held-out 27.6% — held-out lines (no true founder in the index at all) show
the *least* PAV recovery, not the most, because they place fewer reads
overall to begin with (see the placement-rate finding in the original eval).

Full per-individual table: `results/refbias_pav_fraction.tsv` (80 rows, 40
individuals x 2 chain arms).

**Derivation, not measurement — and exact.** PAV fraction equals
`1 - hit_ratio["B73"]` for chain, because (proven in Q2 below) every
reference-anchored row is guaranteed to contain B73 and every PAV row is
guaranteed not to. Verified against two directly-measured `chain_nolift`
(no `--lift`, no PAV path at all) baselines cached from an earlier session
(`scratch/simval_eval/{OUT-INBRED__Tx303,IDX-INBRED__Oh43}__0.1x__chain_nolift`):
the derived and directly-measured PAV fractions agree to floating-point
precision (diff < 1e-16) on all 4 checks (2 individuals x 2 `--pav-agree`
settings).

## Question 2: why is B73's share higher than expected?

**Root cause, proven at the source level** (`search.c`,
`pav-ps4g-insertion-rows` worktree):

1. **`chain_emit()` (reference-anchored path), `search.c:1318-1342`.** A SMEM
   only becomes a chaining candidate if it has **≥1 occurrence in B73**
   (`nref == 0` -> discarded, line 1336) — any assembly-only SMEM from the
   *same read* is silently thrown away, even if it's the longer, more
   discriminating segment. The row's gameteSet is the **intersection**
   across every admitted SMEM's occurrence set (`search.c:1389-1392`).
   Since every admitted SMEM contains B73 by construction, **B73 survives
   that intersection unconditionally, on 100% of reference-anchored rows** —
   regardless of how divergent the rest of the read actually is from B73.
2. **`chain_emit_pav()` (PAV path), `search.c:1116-1305`.** The mirror image:
   only SMEMs with **zero** reference occurrences are admitted as PAV
   candidates (line 1135) — B73 is structurally excluded from every PAV
   gameteSet. The downstream majority-vote position projection and
   `--pav-agree`/`--pav-grid` machinery operate purely on coordinates
   (confirmed by reading `lift.c:232-332`) and never touch the gamete set.
3. **`--insertion-rows=ordinary` / `--diverged-rows=ordinary` (our settings)
   change nothing about this.** Confirmed at the source level: the `acc`/`na`
   gameteSet passed to the PS4G accumulator is byte-identical whether a row
   is printed in `pav:`-tagged or "ordinary" format — only the printed
   contig/position/5th-column differ. The mechanism above was always active,
   independent of our row-formatting flags.

**Consequence:** for chain, `hit_ratio["B73"]` (of placed reads) reduces
almost exactly to `(# reference-anchored rows) / (# reference-anchored rows
+ # PAV rows)` — a **path-mixing ratio**, not a sequence-similarity
statistic the way it is for refmap (whose B73 credit is exactly its `EXACT`
rate, a real "this read's match is present in B73" quantity). Verified
exactly on Tx303: 523,180 / 718,804 = 0.727847925, matching the reported
`hit_ratio["B73"]` to 6 decimal places.

**A code-predicted, independently-confirmed check.** The PAV path's
acceptance gates (winning-cluster completeness, `--pav-min-len`,
low-complexity filter, `--pav-agree` dispersion) are strictly more numerous
than the ordinary path's single gate (`n_ref <= 1`), so a *tighter*
`--pav-agree` should shrink the PAV-recovered denominator more, and raise
the apparent B73%. That is exactly what the data shows: mean B73
hit_ratio-of-placed is 70.6% at `--pav-agree=256` vs 66.2% at `=5000` — the
more conservative/precise PAV setting shows the *bigger* apparent bias,
which only makes sense once `hit_ratio`-of-placed is understood as a
path-mixing ratio rather than a similarity measure.

## How much of the ~70% is real vs a denominator artifact

`hit_ratio` (of placed reads) uses a denominator that is itself much smaller
and less representative for chain than for refmap — chain places only
49-58% of reads vs refmap's 78-80% (see the original eval's placement-rate
finding). Renormalizing to **all input reads** instead
(`hit_ratio_of_input`, computed by `refbias_parse.py` from the start but
unused until now):

| | B73 hit_ratio (of PLACED reads) | B73 hit_ratio_of_input (of ALL reads) |
|---|---:|---:|
| refmap | 38.5% (35.4-42.1%) | 30.4% (27.9-33.4%) |
| chain_ordinary_5000 | 66.2% (60.4-71.1%) | 34.7% (31.5-38.5%) |
| chain_ordinary_256 | 70.6% (64.7-75.4%) | 34.7% (31.5-38.5%) |

**Most of the "70%+ looks impossible" alarm dissolves once measured against
the right denominator.** 34.7% is comfortably under the ~60% ceiling implied
by ~40% inter-line PAV divergence, and close to refmap's own 30.4%. A
smaller, real gap remains (~4.3 points here; the mash-relatedness-corrected
residual from the original eval, computed on the of-placed metric, was
larger at +0.19-0.23 — the two numbers aren't the same metric, but agree in
direction and in "real but modest, not the ~30-point gap the raw hit_ratio
suggested").

**Assembly completeness is a real but secondary contributor to that
residual, not the primary cause.** B73 is uniquely gapless in the v2 index
(12 sequences, 0 scaffolds) vs 282-1,682 sequences and 1.0-6.9% scaffold
content for the other 24 founders — a deliberate artifact of v1->v2 index
construction (v1 B73 had 685 scaffolds, unremarkable among the panel), not
underlying biology. Regressing background hit_ratio on scaffold count across
the 24 non-B73 founders and extrapolating to zero scaffolds: on the
of-placed metric this explains only ~13% of B73's excess (predicted 0.48 at
zero fragmentation vs actual 0.706, residual +0.224); redone on the
correctly-normalized `hit_ratio_of_input`, the unexplained residual shrinks
further to +0.087-0.097 — about 60% smaller, most of the "mystery" already
resolved by the denominator correction above. Critically, **refmap on the
identical index shows ~0 residual after the same correction** (-0.008 to
-0.010) — ruling out "it's just index structure" as the primary
explanation. The excess is chain-algorithm-specific, matching mechanism #1
above, not an index artifact.

## What this changes about the original eval

The qualitative conclusion of `results/refbias.md` is unchanged: chain shows
real reference bias that refmap does not, confirmed five independent ways.
What changes is the **interpretation of chain's raw magnitude**: the
headline 65-75% numbers substantially overstate the true sequence-sharing
comparison because chain's `hit_ratio` conflates two different things (a
reference-anchored-vs-PAV path-mixing ratio, and a smaller real bias) that
look identical in the metric but have very different explanations. The
`hit_ratio_of_input` renormalization above is the fairer number for any
future "does chain over-credit B73" question.

## Not pursued (flagged, not built)

A minimal code-level fix was identified but not implemented: admit
`nref == 0` SMEMs as chain candidates too at `search.c:1318-1342` (they
can't supply a reference position for the colinear DP, but could still
participate in the gameteSet intersection) — this would let the
intersection actually exclude B73 when a read carries real assembly-only
divergent evidence, likely closing most of the remaining gap. Investigation
only; no source changes made.
