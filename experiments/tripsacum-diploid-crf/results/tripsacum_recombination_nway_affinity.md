# N-way recombination Tripsacum baseline (affinity diploid CRF) -- 5 seeds x {4,6} founders, averaged

Generalizes the 2-way recombination sweep (`tripsacum_recombination.md`) from exactly 2 founders (strict A/B alternation) to a pool of N=4 or N=6 founders -- every distinct individual used anywhere in the pair-based tests, recombined N-way instead of 2-way. H1 and H2 each independently draw a new founder from the pool at each breakpoint (constrained only to differ from the immediately preceding segment), so the true pair at any site can be any 2 of the N founders (or the same one twice -- homozygous). Founder feature columns are masked to the locally-true pair before scoring with the **affinity-conditioned** diploid GRITS-CRF (`checkpoints/diploid-affinity-sim512-h3`). 5 independent random breakpoint/founder-assignment draws are averaged per (founder-set, level).

## Overall (averaged across all levels and seeds, per founder-set size)

| n_founders | n_runs | pair_acc_mean | pair_acc_std | hap_acc_mean |
| --- | --- | --- | --- | --- |
| 4.0 | 20.0 | 0.9722 | 0.0057 | 0.9828 |
| 6.0 | 20.0 | 0.8748 | 0.0348 | 0.9350 |

## Per (founder-set, level)

| n_founders | founder_set | n_breakpoints_per_chrom | n_seeds | het_frac_mean | pair_acc_mean | pair_acc_std | hap_acc_mean | hap_acc_std | homo_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 1 | 5 | 0.6831 | 0.9712 | 0.0088 | 0.9823 | 0.0044 | 0.3028 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 2 | 5 | 0.7734 | 0.9760 | 0.0051 | 0.9847 | 0.0026 | 0.2197 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 5 | 5 | 0.7325 | 0.9700 | 0.0043 | 0.9817 | 0.0022 | 0.2533 |
| 4 | C009-T009+C011-T007+C027-T007+C050-T007 | 10 | 5 | 0.7534 | 0.9715 | 0.0022 | 0.9824 | 0.0011 | 0.2369 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 1 | 5 | 0.8085 | 0.8751 | 0.0512 | 0.9352 | 0.0256 | 0.0997 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 2 | 5 | 0.8337 | 0.8865 | 0.0399 | 0.9409 | 0.0200 | 0.0910 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 5 | 5 | 0.8309 | 0.8798 | 0.0220 | 0.9375 | 0.0110 | 0.0825 |
| 6 | C009-T009+C011-T007+C027-T007+C050-T007+C076-T198+C081-T199 | 10 | 5 | 0.7996 | 0.8577 | 0.0217 | 0.9264 | 0.0109 | 0.0935 |

Full per-seed detail: `tripsacum_recombination_nway_affinity_detail.tsv`.

## Summary

The affinity model shows the **opposite** trend from the plain model as founder-pool size
grows: `pair_acc` N=2 (`tripsacum_recombination_affinity.md`, overall 0.984-0.990) -> N=4
(0.972) -> N=6 (0.875) -- accuracy *falls* as N grows, not rises.

This is consistent with, and further evidence for, the explanation already recorded in
`tripsacum_affinity_comparison.md`: the affinity signal is a genome-wide "which founders
are even possible here" prior, and it gets a cleaner, stronger read the fewer true founders
an individual actually has. N=2 is the most restrictive case (biggest gain there, +40pp over
plain); N=6 is the closest to the generic training distribution's own average founder count
(13.8) and correspondingly the smallest edge over the plain model at that N (0.875 vs 0.824,
only +5.1pp, compared to +40pp at N=2 and +22.5pp at N=4). The affinity model still beats
plain at every N tested here, but the *margin* is shrinking exactly as that hypothesis
predicts -- worth checking whether it continues to shrink (and where, if anywhere, it
crosses back to parity or below) at N=8+ before treating "affinity always wins" as settled.
`homo_pred` is also markedly higher here than the plain model's (~0.08-0.30 vs ~0.01-0.02) --
the affinity model calls homozygosity far more often at higher N, plausibly overcorrecting
as the founder-identification task gets harder, though this isn't confirmed as the cause of
the accuracy drop, just a co-occurring pattern.

