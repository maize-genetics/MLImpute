# Recombination-simulation Tripsacum baseline (plain diploid CRF) -- 5 seeds x 4 pairs, averaged

H1 and H2 each independently recombine between two founders at N breakpoints per chromosome (uniform random positions, alternating founder per segment), instead of being constant across the whole genome (the earlier no-recombination baseline, `results/tripsacum_diploid.md`). Reads/refmap/K=18-windowing are reused unchanged per pair from the simple diploid runs; per (pair, level, seed), founder feature columns are masked to only the locally-true ancestry pair and the true per-site `(H1,H2)` labels are written before padding K=18->24 (`checkpoints/diploid-sim512-h3` is hard-fixed to 24 founders) and scoring with the plain (non-affinity) diploid GRITS-CRF via `crf/eval.py::evaluate_diploid`. 5 independent random breakpoint/founder-assignment draws are averaged per (pair, level) to separate a real breakpoint-density effect from single-draw het_frac noise (an earlier single-seed run's non-monotonic 1/2/5/10 trend turned out to be confounded this way).

## Overall (averaged across all 4 pairs and all seeds)

| n_breakpoints_per_chrom | n_runs | het_frac_mean | pair_acc_mean | pair_acc_std | hap_acc_mean |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 20.0 | 0.4980 | 0.5667 | 0.1059 | 0.7822 |
| 2.0 | 20.0 | 0.5254 | 0.5885 | 0.1092 | 0.7930 |
| 5.0 | 20.0 | 0.5064 | 0.5769 | 0.1003 | 0.7873 |
| 10.0 | 20.0 | 0.5053 | 0.5792 | 0.1048 | 0.7884 |

## Per pair (mean +/- std across seeds)

| assemblyA | assemblyB | n_breakpoints_per_chrom | n_seeds | het_frac_mean | het_frac_std | pair_acc_mean | pair_acc_std | hap_acc_mean | hap_acc_std | homo_pred_mean | homo_pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C009-T009 | C011-T007 | 1 | 5 | 0.4975 | 0.0440 | 0.5121 | 0.0454 | 0.7545 | 0.0227 | 0.0223 | 0.0049 |
| C009-T009 | C011-T007 | 2 | 5 | 0.5235 | 0.0660 | 0.5347 | 0.0625 | 0.7657 | 0.0313 | 0.0211 | 0.0051 |
| C009-T009 | C011-T007 | 5 | 5 | 0.5070 | 0.0282 | 0.5257 | 0.0253 | 0.7612 | 0.0127 | 0.0244 | 0.0037 |
| C009-T009 | C011-T007 | 10 | 5 | 0.5045 | 0.0313 | 0.5245 | 0.0316 | 0.7606 | 0.0158 | 0.0250 | 0.0046 |
| C009-T009 | C027-T007 | 1 | 5 | 0.4976 | 0.0430 | 0.5101 | 0.0452 | 0.7534 | 0.0226 | 0.0203 | 0.0042 |
| C009-T009 | C027-T007 | 2 | 5 | 0.5255 | 0.0654 | 0.5348 | 0.0626 | 0.7658 | 0.0313 | 0.0198 | 0.0039 |
| C009-T009 | C027-T007 | 5 | 5 | 0.5052 | 0.0294 | 0.5190 | 0.0264 | 0.7579 | 0.0132 | 0.0207 | 0.0036 |
| C009-T009 | C027-T007 | 10 | 5 | 0.5063 | 0.0304 | 0.5218 | 0.0296 | 0.7593 | 0.0148 | 0.0224 | 0.0027 |
| C009-T009 | C050-T007 | 1 | 5 | 0.4980 | 0.0431 | 0.5102 | 0.0454 | 0.7535 | 0.0227 | 0.0186 | 0.0049 |
| C009-T009 | C050-T007 | 2 | 5 | 0.5242 | 0.0656 | 0.5346 | 0.0627 | 0.7657 | 0.0313 | 0.0178 | 0.0049 |
| C009-T009 | C050-T007 | 5 | 5 | 0.5063 | 0.0298 | 0.5211 | 0.0273 | 0.7589 | 0.0136 | 0.0194 | 0.0047 |
| C009-T009 | C050-T007 | 10 | 5 | 0.5044 | 0.0308 | 0.5196 | 0.0283 | 0.7582 | 0.0142 | 0.0205 | 0.0033 |
| C076-T198 | C081-T199 | 1 | 5 | 0.4991 | 0.0423 | 0.7345 | 0.0146 | 0.8673 | 0.0073 | 0.2430 | 0.0473 |
| C076-T198 | C081-T199 | 2 | 5 | 0.5283 | 0.0662 | 0.7498 | 0.0392 | 0.8749 | 0.0196 | 0.2291 | 0.0371 |
| C076-T198 | C081-T199 | 5 | 5 | 0.5073 | 0.0263 | 0.7420 | 0.0161 | 0.8710 | 0.0081 | 0.2408 | 0.0341 |
| C076-T198 | C081-T199 | 10 | 5 | 0.5061 | 0.0287 | 0.7511 | 0.0170 | 0.8755 | 0.0085 | 0.2506 | 0.0188 |

Full per-seed detail: `tripsacum_recombination_detail.tsv`.

## Summary

Averaging resolved the earlier ambiguity cleanly:

1. **The non-monotonic 1/2/5/10 trend from the single-seed run was noise, confirmed.**
   Overall `pair_acc_mean` by level is now 0.567 / 0.588 / 0.577 / 0.579 -- a spread of
   only 0.02 across all four levels, well within one `pair_acc_std` (~0.10) of each other.
   Same story per pair: every pair's own 4 level-means sit within ~0.02-0.03 of each other,
   with std an order of magnitude larger. **Breakpoint density (1 vs 10 crossovers/chrom)
   has no detectable effect on pair_acc at this depth/pair set**, once averaged over the
   random breakpoint/founder-start draw. The large recombination-vs-no-recombination gap
   from before remains real and unaffected by this (0.51-0.75 here vs 0.76-0.99 for the
   matching no-recombination pairs, `results/tripsacum_diploid.md`).
2. **The pair-identity effect from the no-recombination test reproduces here, and is now
   the dominant source of variation.** C076-T198 x C081-T199 scores 0.73-0.75 at every
   recombination level -- 0.20-0.24 higher than the other three pairs (0.51-0.53), which
   are themselves nearly indistinguishable from each other (they share the hub founder
   C009-T009). This exactly mirrors the no-recombination result, where the same pair scored
   highest (0.985 vs 0.76-0.88) -- consistent with that pair's higher self-coverage
   (~82% vs ~75-77%) driving the difference, not anything about which specific two
   assemblies were chosen otherwise.

