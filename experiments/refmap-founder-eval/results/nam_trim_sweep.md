# NAM founder-recovery: read-trimming sweep

Quality-trimmed the same 1,000,000-read subsample used in the no-trim baseline (`results/nam_baseline.md`) with `scripts/trim_reads.py` -- end-trim leading/trailing bases below a Phred threshold, drop the read if what remains is shorter than a minimum length (internal low-quality runs are left untouched) -- then re-ran the identical refmap -> windowed-CRF -> Viterbi pipeline (`scripts/nam_trim_sweep.py`, driving `nam_baseline.py`'s functions unmodified) across a 3x3 grid: min-qual in [30, 35, 40], min-len in [50, 75, 100].

No-trim baseline (excluding B73): mean viterbi=0.9547, median=0.9887 (`results/nam_baseline_results.tsv`).

## Mean viterbi (excl. B73) by condition

Rows = min-qual, columns = min-len. Baseline (no trim) = 0.9547.

| Q \ L | 50 | 75 | 100 |
| --- | --- | --- | --- |
| 30 | 0.9546 | 0.9546 | 0.9552 |
| 35 | 0.9557 | 0.9554 | 0.9552 |
| 40 | 0.9551 | 0.9557 | 0.9562 |

## Delta vs baseline, sorted best-to-worst

| tag | min_qual | min_len | mean_viterbi | delta_vs_baseline |
| --- | --- | --- | --- | --- |
| q40_l100 | 40 | 100 | 0.9562 | 0.0015 |
| q40_l75 | 40 | 75 | 0.9557 | 0.0010 |
| q35_l50 | 35 | 50 | 0.9557 | 0.0010 |
| q35_l75 | 35 | 75 | 0.9554 | 0.0008 |
| q35_l100 | 35 | 100 | 0.9552 | 0.0005 |
| q30_l100 | 30 | 100 | 0.9552 | 0.0005 |
| q40_l50 | 40 | 50 | 0.9551 | 0.0004 |
| q30_l50 | 30 | 50 | 0.9546 | -0.0001 |
| q30_l75 | 30 | 75 | 0.9546 | -0.0001 |

## Weak-founder detail (CML103, CML52, Tzi8)

| founder | baseline | q30_l50 | q30_l75 | q30_l100 | q35_l50 | q35_l75 | q35_l100 | q40_l50 | q40_l75 | q40_l100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CML103 | 0.7915 | 0.7907 | 0.7907 | 0.7900 | 0.7898 | 0.7902 | 0.7904 | 0.7903 | 0.7913 | 0.7923 |
| CML52 | 0.8898 | 0.8890 | 0.8895 | 0.8981 | 0.8921 | 0.8910 | 0.8876 | 0.8970 | 0.8975 | 0.9005 |
| Tzi8 | 0.5209 | 0.5150 | 0.5150 | 0.5170 | 0.5227 | 0.5209 | 0.5184 | 0.5192 | 0.5243 | 0.5308 |

## Trim aggressiveness by condition

Mean over founders of `pct_kept` (reads surviving trim+min-len) and `mean_len_out` (mean trimmed length, kept reads only) -- the cost side of the accuracy/length tradeoff.

| tag | min_qual | min_len | mean_pct_kept | mean_len_out |
| --- | --- | --- | --- | --- |
| q30_l50 | 30 | 50 | 99.9985 | 149.5178 |
| q30_l75 | 30 | 75 | 99.9941 | 149.5214 |
| q30_l100 | 30 | 100 | 99.9831 | 149.5280 |
| q35_l50 | 35 | 50 | 99.9929 | 146.8467 |
| q35_l75 | 35 | 75 | 99.9760 | 146.8605 |
| q35_l100 | 35 | 100 | 99.9079 | 146.8988 |
| q40_l50 | 40 | 50 | 99.9322 | 142.3501 |
| q40_l75 | 40 | 75 | 99.8157 | 142.4412 |
| q40_l100 | 40 | 100 | 99.3746 | 142.6725 |

