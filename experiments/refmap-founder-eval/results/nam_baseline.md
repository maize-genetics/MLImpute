# NAM-founder full-pipeline baseline

Real WGS reads (one file per NAM founder, `WGSReads/`) run through the same
`ropebwt3 refmap` whole-read placement recipe as the original Oh43 benchmark
(`--ref-prefix=B73 --max-occ=-1 --lift ... -t 20`, no `--kmer*`), on a
1,000,000-read subsample of each founder's file. The subsample is a literal
`head` of the first 1,000,000 reads, matching exactly how the historical
`bench_ps4g_npy/reads/Oh43_1M.fastq` was built (verified byte-identical first
read against the un-subsampled source). `--target-hits=1000000` was tried first
but rejected: it overshoots to ~1.96M EXACT+PLACED records because the
per-thread target check only fires between ~666k-read batches -- the same
overshoot visible in the historical `target1M.log`, which is presumably why the
original benchmark used a pre-subsampled file instead of the flag.

Reads were windowed (`crf/ropebwt_npy_to_matrix.py --window-size=512`,
K=25 panel trimmed to K=24 by dropping the least-covered founder genome-wide)
and scored with the `haploid-sim/last.ckpt` CRF -- the same checkpoint used
throughout the Oh43 investigation -- on the held-out test split (`make_splits`,
val_frac=test_frac=0.10), identical methodology to the existing Oh43 number.

For reference, the existing Oh43 benchmark (older DebugSim-simulated 1x reads,
patched 64-carrier-cap binary) scored `viterbi=0.9915`
(`results/eval.tsv`, `haploid-sim-on-ropebwt-oh43-k24-PATCHED`). The fresh
real-WGS Oh43 row here (`viterbi=0.9880`)
lands in the same ballpark on an independently re-downloaded read set --
validating the whole chain end-to-end on genuinely new data.

## Per-founder results

Sorted by `viterbi` (overall per-site Viterbi-decode accuracy on the held-out
test split); `cov`/`abs` = founder-covered vs founder-absent site split (absent
= the founder's own K24 feature column reads zero at that site -- the model
cannot be right there by construction, mirroring the Oh43 covered/absent
analysis in `oh43_absent_rootcause.md`).

| founder | n_placed | n_unplaced | mean_founders_per_site | self_coverage_pct_k25 | n_sites | viterbi | cov_pct | cov_acc | abs_pct | abs_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ki11 | 724796 | 21479 | 9.7632 | 97.8003 | 63488 | 0.9944 | 97.8909 | 0.9951 | 2.1091 | 0.9642 |
| P39 | 756739 | 8234 | 8.5436 | 98.0996 | 65536 | 0.9944 | 98.1689 | 0.9950 | 1.8311 | 0.9617 |
| Ms71 | 730811 | 16790 | 9.7261 | 96.0821 | 67072 | 0.9940 | 97.3446 | 0.9956 | 2.6554 | 0.9326 |
| HP301 | 761061 | 8028 | 9.2606 | 97.9533 | 65536 | 0.9939 | 97.9919 | 0.9959 | 2.0081 | 0.8944 |
| M37W | 719427 | 13631 | 9.8056 | 97.8886 | 61952 | 0.9938 | 98.0049 | 0.9952 | 1.9951 | 0.9248 |
| B97 | 736176 | 12677 | 9.7412 | 95.8710 | 68096 | 0.9927 | 97.4507 | 0.9961 | 2.5493 | 0.8646 |
| CML277 | 710634 | 18144 | 9.5644 | 97.0225 | 56320 | 0.9910 | 97.3775 | 0.9932 | 2.6225 | 0.9086 |
| CML322 | 695626 | 25329 | 9.5022 | 96.7991 | 54272 | 0.9906 | 97.0574 | 0.9941 | 2.9426 | 0.8766 |
| Mo18W | 732792 | 22387 | 9.3026 | 97.2442 | 62976 | 0.9906 | 97.4038 | 0.9926 | 2.5962 | 0.9119 |
| CML333 | 719185 | 18835 | 9.6692 | 97.2646 | 58880 | 0.9892 | 97.4236 | 0.9908 | 2.5764 | 0.9295 |
| CML247 | 722537 | 20687 | 9.5578 | 96.9702 | 56832 | 0.9889 | 97.3026 | 0.9918 | 2.6974 | 0.8832 |
| NC358 | 705842 | 20881 | 9.7219 | 96.9757 | 55808 | 0.9887 | 97.2567 | 0.9912 | 2.7433 | 0.9020 |
| Oh43 | 736257 | 15734 | 9.5463 | 97.3884 | 67584 | 0.9880 | 97.6400 | 0.9888 | 2.3600 | 0.9524 |
| Ki3 | 697831 | 28317 | 9.8952 | 94.6304 | 64512 | 0.9878 | 94.9436 | 0.9899 | 5.0564 | 0.9491 |
| Oh7B | 737775 | 11068 | 9.9116 | 97.7170 | 63488 | 0.9867 | 97.8390 | 0.9883 | 2.1610 | 0.9155 |
| NC350 | 710739 | 20672 | 9.6978 | 96.4940 | 55808 | 0.9834 | 96.6922 | 0.9864 | 3.3078 | 0.8954 |
| Il14H | 760153 | 6393 | 8.7404 | 97.4985 | 58880 | 0.9823 | 97.5476 | 0.9842 | 2.4524 | 0.9086 |
| CML69 | 723559 | 17908 | 9.5336 | 96.2008 | 56320 | 0.9786 | 96.4595 | 0.9893 | 3.5405 | 0.6876 |
| B73 | 758564 | 5058 | 9.7783 | 97.7380 | 66560 | 0.9758 | 97.8561 | 0.9767 | 2.1439 | 0.9341 |
| CML228 | 701089 | 20117 | 9.7606 | 95.1214 | 56832 | 0.9747 | 96.7184 | 0.9805 | 3.2816 | 0.8032 |
| Ky21 | 748248 | 11586 | 9.6478 | 96.8166 | 59392 | 0.9720 | 97.0973 | 0.9748 | 2.9027 | 0.8765 |
| CML52 | 702311 | 26549 | 9.6350 | 89.0111 | 59392 | 0.8898 | 91.4281 | 0.9306 | 8.5719 | 0.4549 |
| CML103 | 747345 | 9804 | 9.4332 | 95.9680 | 68608 | 0.7915 | 85.7611 | 0.8994 | 14.2389 | 0.1412 |
| Tzi8 | 686188 | 23568 | 9.7371 | 80.1551 | 59904 | 0.5209 | 72.1271 | 0.6612 | 27.8729 | 0.1576 |

## B73 self-mapping sanity check

B73 is the `--ref-prefix` reference itself; its row (`viterbi=0.9758`, `self_coverage_pct_k25=97.7%`) is high and in the same range as the other well-behaved founders, confirming no
labels/index-mapping bug -- excluded from the outlier/summary stats below since
it isn't a generalization test.

## Outliers


3 founder(s) score well below the rest (20 other non-reference founders land in `viterbi` 0.972-0.994):


- **Tzi8**: `viterbi=0.5209`, `self_coverage_pct_k25=80.2%`, `abs_pct=27.9%`, `abs_acc=0.1576`

- **CML103**: `viterbi=0.7915`, `self_coverage_pct_k25=96.0%`, `abs_pct=14.2%`, `abs_acc=0.1412`

- **CML52**: `viterbi=0.8898`, `self_coverage_pct_k25=89.0%`, `abs_pct=8.6%`, `abs_acc=0.4549`

Investigated to rule out a driver bug before reporting these: label BEDs correctly carry each founder's own name (no copy/mapping error), refmap EXACT/PLACED/MULTI/UNPLACED status proportions are comparable to the well-behaved founders (no placement-rate collapse), and mean per-read base quality is uniform across all founders (Q39-40, no quality-driven explanation).

**Tzi8** (1,548 contigs) and **CML52** (1,682 contigs) have by far the most fragmented assemblies of those checked (630-760 contigs for B73/Oh43/CML103), plausibly explaining their low self-coverage via the same repeat/paralog-mismapping and assembly-gap mechanisms documented for the Oh43 residual (`results/oh43_residual_4pct.md`), at much larger scale. **CML103** breaks this pattern (632 contigs, among the *least* fragmented) yet still scores worst of all -- its cause is unresolved and flagged for follow-up, not root-caused here (out of scope for this baseline run).


## Excluded founders


- **Tx303**: WGS read file present in `WGSReads/`, but no matching assembly/panel column in this pangenome index -- no valid truth column, not scored.
- **M162W**: present in the 25-founder pangenome panel, but no WGS read file was downloaded -- cannot be run.


## Summary (excluding B73)


- N founders scored: 23

- viterbi mean=0.9547  median=0.9887  min=0.5209  max=0.9944

