# Tier 1 evaluation: before/after on real 0.1x rows

Baseline: `checkpoints/diploid-affinity-sim512-h3/` (fixed homo_penalty, no learned-het).
New: `checkpoints/breedpop-sparse-affinity-learnedhet/` (learned-het + founder-affinity, sparse-crossover breeding-pop training data).

| row | metric | baseline | new | delta |
|---|---|---|---|---|
| IDX-INBRED__Oh43__0.1x | pair_acc | 1.0000 | 0.8400 | -0.1600 |
| IDX-INBRED__Oh43__0.1x | hap_acc | 1.0000 | 0.9200 | -0.0800 |
| IDX-INBRED__Oh43__0.1x | marginal_hap_acc | 0.5000 | 0.5000 | +0.0000 |
| IDX-INBRED__Oh43__0.1x | mean_decoded_switches | 0.0000 | 0.0000 | +0.0000 |
| IDX-INBRED__Oh43__0.1x | mean_true_switches | 0.0000 | 0.0000 | +0.0000 |
| IDX-INBRED__Oh43__0.1x | het_frac_decoded_pen | 0.0000 | 0.1600 | +0.1600 |
| IDX-INBRED__Il14H__0.1x | pair_acc | 1.0000 | 0.9000 | -0.1000 |
| IDX-INBRED__Il14H__0.1x | hap_acc | 1.0000 | 0.9500 | -0.0500 |
| IDX-INBRED__Il14H__0.1x | marginal_hap_acc | 0.5000 | 0.5000 | +0.0000 |
| IDX-INBRED__Il14H__0.1x | mean_decoded_switches | 0.0000 | 0.0000 | +0.0000 |
| IDX-INBRED__Il14H__0.1x | mean_true_switches | 0.0000 | 0.0000 | +0.0000 |
| IDX-INBRED__Il14H__0.1x | het_frac_decoded_pen | 0.0000 | 0.1000 | +0.1000 |
| IDX-HYB__Oh43xIl14H__0.1x | pair_acc | 0.8281 | 0.6179 | -0.2102 |
| IDX-HYB__Oh43xIl14H__0.1x | hap_acc | 0.8930 | 0.7859 | -0.1071 |
| IDX-HYB__Oh43xIl14H__0.1x | marginal_hap_acc | 0.8920 | 0.8302 | -0.0618 |
| IDX-HYB__Oh43xIl14H__0.1x | mean_decoded_switches | 1.5800 | 0.5100 | -1.0700 |
| IDX-HYB__Oh43xIl14H__0.1x | mean_true_switches | 0.0000 | 0.0000 | +0.0000 |
| IDX-HYB__Oh43xIl14H__0.1x | het_frac_decoded_pen | 0.9968 | 0.7668 | -0.2300 |
| IDX-RIL__Oh43xIl14H__0.1x | pair_acc | 0.5457 | 0.6828 | +0.1371 |
| IDX-RIL__Oh43xIl14H__0.1x | hap_acc | 0.7626 | 0.8350 | +0.0723 |
| IDX-RIL__Oh43xIl14H__0.1x | marginal_hap_acc | 0.7268 | 0.7176 | -0.0092 |
| IDX-RIL__Oh43xIl14H__0.1x | mean_decoded_switches | 2.4800 | 0.2200 | -2.2600 |
| IDX-RIL__Oh43xIl14H__0.1x | mean_true_switches | 0.0500 | 0.0500 | +0.0000 |
| IDX-RIL__Oh43xIl14H__0.1x | het_frac_decoded_pen | 0.8636 | 0.5169 | -0.3468 |
| sim_k2_ind0674 | pair_acc | 0.4539 | 0.6137 | +0.1598 |
| sim_k2_ind0674 | hap_acc | 0.7211 | 0.7994 | +0.0784 |
| sim_k2_ind0674 | marginal_hap_acc | 0.7191 | 0.7118 | -0.0073 |
| sim_k2_ind0674 | mean_decoded_switches | 4.6300 | 1.0000 | -3.6300 |
| sim_k2_ind0674 | mean_true_switches | 5.0800 | 5.0800 | +0.0000 |
| sim_k2_ind0674 | het_frac_decoded_pen | 0.9916 | 0.2996 | -0.6920 |
| sim_k13_ind0222 | pair_acc | 0.6370 | 0.5357 | -0.1013 |
| sim_k13_ind0222 | hap_acc | 0.7727 | 0.7207 | -0.0520 |
| sim_k13_ind0222 | marginal_hap_acc | 0.7871 | 0.7522 | -0.0349 |
| sim_k13_ind0222 | mean_decoded_switches | 3.1400 | 1.6900 | -1.4500 |
| sim_k13_ind0222 | mean_true_switches | 4.9600 | 4.9600 | +0.0000 |
| sim_k13_ind0222 | het_frac_decoded_pen | 0.9989 | 0.7727 | -0.2262 |

## RIL homozygous vs heterozygous truth split (the H3-per-locus test)

| checkpoint | homo-truth windows n | homo_acc | het-truth windows n | het_acc | gap |
|---|---|---|---|---|---|
| diploid-affinity-sim512-h3 | 43 | 0.2179 | 57 | 0.7929 | -0.5750 |
| breedpop-sparse-affinity-learnedhet | 43 | 0.7298 | 57 | 0.6473 | +0.0825 |
