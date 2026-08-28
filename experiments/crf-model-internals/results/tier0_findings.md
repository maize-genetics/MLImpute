# Tier 0 findings + RIL mechanism confirmation

Companion to `results/support_profile.md` and
`/home/zrm22/.claude/plans/wondrous-discovering-octopus.md`. All results
below use the deployed `checkpoints/diploid-affinity-sim512-h3/
d-epoch=04-val_pair_acc=0.6179.ckpt` (fixed `homo_penalty=3.0`,
`learned_het=False`) unless noted.

## Tier 0a — homo-penalty sensitivity sweep (`scripts/homo_penalty_sweep.py`)

Recomputed `emis_p` at pen in {0, 0.5, ..., 4.0} from the already-dumped
`emis_f`, re-decoded with `_dcrf_viterbi`, on the real Oh43/Il14H/Oh43xIl14H
0.1x rows (24 windows each).

| pen | hybrid pair_acc | inbred (Oh43) pair_acc | inbred (Il14H) pair_acc |
|---|---|---|---|
| 0.0 | 0.000 | 1.000 | 1.000 |
| 0.5 | 0.766 | 0.497 | 0.759 |
| 1.0 | 0.843 | 0.174 | 0.403 |
| 2.0 | 0.870 | 0.071 | 0.203 |
| 3.0 (production) | 0.884 | 0.042 | 0.141 |
| 4.0 | 0.896 | 0.025 | 0.105 |

**No fixed penalty serves both** — hybrid keeps climbing through pen=4.0
while inbred collapses the instant any penalty is applied. A real-data
replay of E7's original finding. Production's per-kind `homo_scale`
multiplier already places each row at its own correct end of this trade-off
(inbred: `homo_scale=0` → pen=0 always, regardless of `homo_penalty`'s
value). One free, zero-risk tweak this reveals: raising the checkpoint's
global `homo_penalty` constant (3.0→~4.0) would gain hybrid/RIL a further
~1pp without touching inbred at all (inbred's pen is always `0 × anything =
0`). Not applied — noted for later, since Tier 1 supersedes it.

## Tier 0b — whole-run vs per-window decode (`scripts/whole_run_decode_check.py`)

24 genuinely adjacent windows (not last session's genome-spread sample) from
the real hybrid row, decoded two ways: (a) independent per-512-site-window
Viterbi (what production does), (b) one Viterbi call over the whole
12,288-site concatenated run.

| | pair_acc | within-window spurious switches |
|---|---|---|
| (a) per-window | 0.897 | 35 |
| (b) whole-run | 0.920 | 37 |

Confirms `docs/RESULTS.md`'s **E12-edge** finding transfers to real data:
whole-run decoding gives a small (+2.2pp) accuracy gain from boundary
context, but does **not** reduce the spurious switch rate itself. Rules out
decode-scope as the H2 fix.

## RIL mechanism — measured directly, not just hypothesized

User's challenge before approving Tier 1: RIL has *more* recombination than
hybrid (genuinely mosaic), so if H2 alone explained things RIL should score
*better* than hybrid, not worse — but it's worse (`simval_results.tsv`: RIL
7.92% error vs hybrid 7.31%). Traced to the corpus's construction
(`config.py:165`: 40 breakpoints/haplotype genome-wide, independent for h1
and h2) → ~52Mb average segment, so RIL is just as locally-constant per
window as hybrid; H2 doesn't distinguish them.

Wired up real per-site RIL truth (`_window_ril_truth` in
`dump_model_internals.py`, via `simval_truth_labels.bin_truth_labels` →
`simval_oracle_bed.build_ril_mosaics`) and dumped
`IDX-RIL__Oh43xIl14H__0.1x` (24 windows) on the deployed checkpoint.
Self-check exact (`max_diff=0.0`). Truth sanity: strictly `{Oh43, Il14H}` at
every site (no third founder ever appears), only 1/24 sampled windows has an
internal breakpoint (matches the ~52Mb-segment prediction), 11/24 windows
genuinely homozygous (matches the ~50% construction).

**Per-window pair_acc splits cleanly by true local zygosity:**

| | n windows | mean pair_acc |
|---|---|---|
| homozygous-truth windows | 11 | **0.223** |
| heterozygous-truth windows | 13 | **0.867** |

A 64-point gap, exactly as predicted: the fixed `homo_scale=0.5` (a genome-
wide compromise between the true 0 and true 1 that any given RIL block
actually needs) systematically fails on homozygous blocks specifically —
consistent with the model defaulting toward "heterozygous" under a
compromise penalty. This is the real-data confirmation of the mechanism
described in the plan (RIL as a within-genome instance of E7's "fixed prior
can't serve varying truth" finding) — not previously measured, now is.
**Prediction for Tier 1: `--learned-het` should close most of this
64-point homo/het split, and help RIL at least as much as hybrid.**

## Status

Tier 0 complete. Tier 1 (data generation + retrain) in progress —
`data/training/sim_breedpop_sparse.npy` generated and verified (class mix
23/24/53% ≈ target, crossovers/window mean=0.50 min=0 max=1, 56% of sampled
individuals now have genuine het_frac<0.05 vs 0 before). Training running as
`checkpoints/breedpop-sparse-affinity-learnedhet/`. Evaluation against the
same real rows (including this RIL homo/het split) pending training
completion.
