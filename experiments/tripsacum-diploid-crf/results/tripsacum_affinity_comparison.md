# Plain vs. affinity diploid CRF on the Tripsacum pipelines: comparison

Both `scripts/tripsacum_diploid.py` and `scripts/tripsacum_recombination.py` were extended
to support the affinity-conditioned checkpoint (`checkpoints/diploid-affinity-sim512-h3`,
trained earlier this session), computing a genome-wide `_founder_affinity` vector from each
run's own data and feeding it as `ext_emb` (`tripsacum_diploid.
evaluate_diploid_with_affinity` — the diploid equivalent of the "single real individual, no
`--windows-per-individual` grouping" pattern `crf/eval.py` already documents and implements
for haploid mode, but never wires up for diploid). All reads/refmap/windowing/masking/
padding were reused unchanged from the existing plain-model runs — only the final scoring
step is new.

## Result: a much larger gain here than on the synthetic held-out test split

The training-time evaluation (`results/diploid_affinity_training.md`) found a modest,
consistent **~2-2.5 percentage point** `hap_acc` gain on `simulate_alleles.py`'s generic
synthetic distribution (each individual restricted to a *random* subset of 2-24 founders,
mean 13.8). **On the Tripsacum pipelines, the gain is dramatically larger.** The reason is
structural, not a fluke: our controlled tests always restrict a synthetic individual to
**exactly 2** founders genome-wide (by construction — that's the whole "combine two
assemblies' reads" design) — a far stronger, cleaner "which founders are even possible here"
signal than the generic 2-24-founder training distribution the model saw. The affinity prior
is exactly suited to this case.

### No-recombination test (`tripsacum_diploid.py`, 4 pairs)

| Pair | plain `pair_acc` | affinity `pair_acc` | Δ |
|---|---|---|---|
| C009-T009 x C011-T007 | 0.763 | 0.982 | +21.9 pp |
| C009-T009 x C027-T007 | 0.881 | 0.989 | +10.8 pp |
| C009-T009 x C050-T007 | 0.879 | 0.989 | +11.0 pp |
| C076-T198 x C081-T199 | 0.985 | 0.995 | +1.0 pp |

The largest gains land on the pairs that were previously *worst* (the three
"C009-T009 x hub-founder" pairs, ~0.76-0.88 plain) — the affinity signal disproportionately
helps the harder cases, not just the already-easy one (which was already near ceiling and had
little room to improve).

### Recombination sweep (`tripsacum_recombination.py`, 4 pairs x 4 levels x 5 seeds, averaged)

| Breakpoints/chrom | plain `pair_acc` (mean ± std) | affinity `pair_acc` (mean ± std) | Δ |
|---|---|---|---|
| 1 | 0.567 ± 0.106 | 0.990 ± 0.002 | +42.3 pp |
| 2 | 0.588 ± 0.109 | 0.989 ± 0.002 | +40.1 pp |
| 5 | 0.577 ± 0.100 | 0.987 ± 0.003 | +41.0 pp |
| 10 | 0.579 ± 0.105 | 0.984 ± 0.003 | +40.5 pp |

Two further findings only visible with the affinity model:

1. **A real breakpoint-density effect now exists** where none was detectable before. The
   plain model's level-to-level differences (std ~0.10) were pure noise (see
   `tripsacum_recombination.md`'s summary). The affinity model's std collapsed to ~0.002-0.003
   (~40x smaller) — small enough that the monotonic decline from 1 to 10 breakpoints
   (0.990 -> 0.989 -> 0.987 -> 0.984, a 0.6pp spread across levels whose 1-seed std is
   ~0.002-0.003) is now a real, if modest, signal: more recombination *is* harder to decode,
   as expected, but this was completely masked by the plain model's noise floor.
2. **The pair-identity gap has nearly closed.** Plain model: the best pair (C076-T198 x
   C081-T199, higher self-coverage) beat the other three by ~20-24 pp at every level. Affinity
   model: that gap shrinks to ~0.5-1.5 pp (e.g. at 10 breakpoints: 0.988 vs 0.982-0.983).
   `homo_pred` also tracks the true homozygous fraction closely now (e.g. mean 0.500 vs true
   homozygosity ~0.50 at level 1 across pairs) — the model isn't just scoring higher, it's
   also much better calibrated.

## Bottom line

The affinity conditioning that gave a modest ~2-2.5pp gain on the generic synthetic
evaluation gives a **~10-42 percentage point** gain on our Tripsacum tests specifically,
because our tests' controlled 2-founder-genome-wide structure is an unusually clean match
for what the affinity prior detects. This is a much stronger validation of the mechanism
than the training-time number alone suggested — worth remembering that the size of the
affinity benefit is highly dependent on how restricted the true founder set is per
individual, not a fixed property of the model.

Full per-pair/per-level detail: `tripsacum_diploid_affinity.tsv`/`.md`,
`tripsacum_recombination_affinity_detail.tsv`/`.md`.
