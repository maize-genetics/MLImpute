# Zero-sequencing-error control: IDX-HYB / Oh43xIl14H / 0.1x

## All-sites and SNP+RefCall error rate by arm

| arm | error_rate | partial_error_rate | snprc_error_rate | compared_sites |
|---|---|---|---|---|
| A_baseline | 7.3138% | 3.6571% | 4.1433% | 156353772 |
| B_noerror | 7.8543% | 3.9272% | 4.5261% | 156289530 |
| C_redraw | 7.1697% | 3.5848% | 3.9515% | 156306957 |

## PS4G aggregate background

| arm | total_unique_counts | mean_true_parent_count | median_nonparent_count | background_ratio |
|---|---|---|---|---|
| A_baseline | 1151382 | 776249.0 | 372171.0 | 0.479 |
| B_noerror | 1158280 | 786252.0 | 370262.0 | 0.471 |
| C_redraw | 1153679 | 777579.0 | 371060.0 | 0.477 |

## L3 true-source x credited-founder confusion, B_noerror

status counts (all reads, labeled+unlabeled): {'EXACT': 443156, 'PLACED': 715124, 'MULTI': 264929, 'UNPLACED': 10443}

| true source | n_reads | self_credit_rate (of PLACED) | mean_other_founder_credit | background_ratio |
|---|---|---|---|---|
| Oh43 | 725472 | 15.301% | 82021.2 | 0.241 |
| Il14H | 708180 | 17.396% | 77282.0 | 0.206 |

## L3 true-source x credited-founder confusion, C_redraw

status counts (all reads, labeled+unlabeled): {'PLACED': 711124, 'MULTI': 268991, 'EXACT': 442555, 'UNPLACED': 10982}

| true source | n_reads | self_credit_rate (of PLACED) | mean_other_founder_credit | background_ratio |
|---|---|---|---|---|
| Oh43 | 725472 | 15.113% | 81528.6 | 0.244 |
| Il14H | 708180 | 17.119% | 77036.2 | 0.210 |
