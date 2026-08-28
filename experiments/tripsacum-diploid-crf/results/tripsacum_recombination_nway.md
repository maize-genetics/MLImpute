# N-way recombination Tripsacum baseline (plain diploid CRF) -- 5 seeds x {4,6} founders, averaged

Generalizes the 2-way recombination sweep (`tripsacum_recombination.md`) from exactly 2 founders (strict A/B alternation) to a pool of N=4 or N=6 founders -- every distinct individual used anywhere in the pair-based tests, recombined N-way instead of 2-way. H1 and H2 each independently draw a new founder from the pool at each breakpoint (constrained only to differ from the immediately preceding segment), so the true pair at any site can be any 2 of the N founders (or the same one twice -- homozygous). Founder feature columns are masked to the locally-true pair before scoring with the plain (non-affinity) diploid GRITS-CRF. 5 independent random breakpoint/founder-assignment draws are averaged per (founder-set, level).

## Overall (averaged across all levels and seeds, per founder-set size)

| n_founders | n_runs | pair_acc_mean | pair_acc_std | hap_acc_mean |
| --- | --- | --- | --- | --- |
| 4.0 | 20.0 | 0.7475 | 0.0612 | 0.8704 |
| 6.0 | 20.0 | 0.8244 | 0.0394 | 0.9098 |

## Per (founder-set, level)

| n_founders | founder_set | n_breakpoints_per_chrom | n_seeds | het_frac_mean | pair_acc_mean | pair_acc_std | hap_acc_mean | hap_acc_std | homo_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 1 | 5 | 0.6831 | 0.6938 | 0.0938 | 0.8436 | 0.0469 | 0.0168 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 2 | 5 | 0.7734 | 0.7856 | 0.0290 | 0.8895 | 0.0145 | 0.0187 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 5 | 5 | 0.7325 | 0.7479 | 0.0428 | 0.8706 | 0.0214 | 0.0212 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 10 | 5 | 0.7534 | 0.7626 | 0.0246 | 0.8779 | 0.0123 | 0.0163 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 1 | 5 | 0.8085 | 0.8149 | 0.0464 | 0.9050 | 0.0232 | 0.0085 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 2 | 5 | 0.8337 | 0.8398 | 0.0474 | 0.9175 | 0.0237 | 0.0080 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 5 | 5 | 0.8309 | 0.8370 | 0.0292 | 0.9160 | 0.0146 | 0.0092 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 10 | 5 | 0.7996 | 0.8060 | 0.0327 | 0.9005 | 0.0163 | 0.0099 |

Full per-seed detail: `tripsacum_recombination_nway_detail.tsv`.

## Summary

Counter to the naive "more founders = harder" expectation, `pair_acc` **increases** with
founder-pool size for the plain model: N=2 (`tripsacum_recombination.md`, overall
0.567-0.588) -> N=4 (0.747) -> N=6 (0.824). `het_frac` also climbs with N (0.50 -> ~0.68-0.77
-> ~0.80-0.83) -- mechanically expected, since two independent draws from a larger pool are
less likely to coincidentally land on the same founder. Given `homo_pred` stays low and
roughly flat across all three (the plain model is consistently biased toward predicting
heterozygous), a higher true-het rate plays to that existing bias rather than against it,
which plausibly drives the higher `pair_acc` -- not confirmed as the full explanation, just
the most consistent read of what's here. See `tripsacum_recombination_nway_affinity.md` for
a very different pattern with the affinity model, and a check of whether that pattern
generalizes further would need N=8+/more founder-set draws.

