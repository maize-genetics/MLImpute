# Recombination-simulation Tripsacum baseline (affinity diploid CRF) -- 5 seeds x 4 pairs, averaged

H1 and H2 each independently recombine between two founders at N breakpoints per chromosome (uniform random positions, alternating founder per segment), instead of being constant across the whole genome (the earlier no-recombination baseline, `results/tripsacum_diploid.md`). Reads/refmap/K=18-windowing are reused unchanged per pair from the simple diploid runs; per (pair, level, seed), founder feature columns are masked to only the locally-true ancestry pair and the true per-site `(H1,H2)` labels are written before padding K=18->24 (the checkpoint is hard-fixed to 24 founders) and scoring with the **affinity-conditioned** diploid GRITS-CRF (`checkpoints/diploid-affinity-sim512-h3`), fed a genome-wide `_founder_affinity` vector computed from each run's own data (`tripsacum_diploid.evaluate_diploid_with_affinity`). 5 independent random breakpoint/founder-assignment draws are averaged per (pair, level) to separate a real breakpoint-density effect from single-draw het_frac noise (an earlier single-seed run's non-monotonic 1/2/5/10 trend turned out to be confounded this way).

## Overall (averaged across all 4 pairs and all seeds)

| n_breakpoints_per_chrom | n_runs | het_frac_mean | pair_acc_mean | pair_acc_std | hap_acc_mean |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 20.0 | 0.4980 | 0.9896 | 0.0024 | 0.9936 |
| 2.0 | 20.0 | 0.5254 | 0.9887 | 0.0021 | 0.9931 |
| 5.0 | 20.0 | 0.5064 | 0.9869 | 0.0030 | 0.9922 |
| 10.0 | 20.0 | 0.5053 | 0.9840 | 0.0029 | 0.9908 |

## Per pair (mean +/- std across seeds)

| assemblyA | assemblyB | n_breakpoints_per_chrom | n_seeds | het_frac_mean | het_frac_std | pair_acc_mean | pair_acc_std | hap_acc_mean | hap_acc_std | homo_pred_mean | homo_pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C009-T009 | C011-T007 | 1 | 5 | 0.4975 | 0.0440 | 0.9886 | 0.0011 | 0.9927 | 0.0005 | 0.5002 | 0.0431 |
| C009-T009 | C011-T007 | 2 | 5 | 0.5235 | 0.0660 | 0.9876 | 0.0007 | 0.9922 | 0.0003 | 0.4756 | 0.0671 |
| C009-T009 | C011-T007 | 5 | 5 | 0.5070 | 0.0282 | 0.9857 | 0.0016 | 0.9913 | 0.0008 | 0.4872 | 0.0263 |
| C009-T009 | C011-T007 | 10 | 5 | 0.5045 | 0.0313 | 0.9827 | 0.0014 | 0.9898 | 0.0007 | 0.4885 | 0.0315 |
| C009-T009 | C027-T007 | 1 | 5 | 0.4976 | 0.0430 | 0.9883 | 0.0014 | 0.9926 | 0.0007 | 0.5003 | 0.0429 |
| C009-T009 | C027-T007 | 2 | 5 | 0.5255 | 0.0654 | 0.9870 | 0.0008 | 0.9919 | 0.0004 | 0.4745 | 0.0664 |
| C009-T009 | C027-T007 | 5 | 5 | 0.5052 | 0.0294 | 0.9849 | 0.0021 | 0.9908 | 0.0010 | 0.4893 | 0.0285 |
| C009-T009 | C027-T007 | 10 | 5 | 0.5063 | 0.0304 | 0.9818 | 0.0013 | 0.9893 | 0.0006 | 0.4861 | 0.0287 |
| C009-T009 | C050-T007 | 1 | 5 | 0.4980 | 0.0431 | 0.9880 | 0.0005 | 0.9924 | 0.0003 | 0.4984 | 0.0434 |
| C009-T009 | C050-T007 | 2 | 5 | 0.5242 | 0.0656 | 0.9882 | 0.0007 | 0.9925 | 0.0003 | 0.4727 | 0.0669 |
| C009-T009 | C050-T007 | 5 | 5 | 0.5063 | 0.0298 | 0.9859 | 0.0020 | 0.9914 | 0.0010 | 0.4864 | 0.0284 |
| C009-T009 | C050-T007 | 10 | 5 | 0.5044 | 0.0308 | 0.9829 | 0.0015 | 0.9898 | 0.0007 | 0.4877 | 0.0293 |
| C076-T198 | C081-T199 | 1 | 5 | 0.4991 | 0.0423 | 0.9934 | 0.0006 | 0.9967 | 0.0003 | 0.5027 | 0.0433 |
| C076-T198 | C081-T199 | 2 | 5 | 0.5283 | 0.0662 | 0.9920 | 0.0010 | 0.9960 | 0.0005 | 0.4722 | 0.0652 |
| C076-T198 | C081-T199 | 5 | 5 | 0.5073 | 0.0263 | 0.9911 | 0.0009 | 0.9955 | 0.0005 | 0.4917 | 0.0257 |
| C076-T198 | C081-T199 | 10 | 5 | 0.5061 | 0.0287 | 0.9884 | 0.0011 | 0.9942 | 0.0006 | 0.4915 | 0.0281 |

Full per-seed detail: `tripsacum_recombination_affinity_detail.tsv`.

