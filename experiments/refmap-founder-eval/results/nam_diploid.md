# Synthetic-diploid baseline (combined real founder reads)

Each row combines two founders' real WGS read files (`WGSReads/`) -- a `head`-based subsample of `depth_per_hap` reads from each, concatenated -- run through the same `ropebwt3 refmap` whole-read placement recipe as `nam_baseline.py` and windowed to K=24 (`crf/ropebwt_npy_to_matrix.py --window-size=512 --target-num-parents=24`). Because the two read sets are combined wholesale (no recombination), truth is exactly `(founderA, founderB)` at every site -- the label columns are overwritten with this known-by-construction pair, and `het_frac` is asserted to be 100%.

Scored with `checkpoints/diploid-sim512-h3/last.ckpt` (`GRITSCRFDiploid`, `crf/eval.py::evaluate_diploid`), the same checkpoint used for the existing `diploid-sim-on-ropebwt-oh43-k24` row in `results/eval.tsv` (a *homozygous* single-founder real-read run, pair_acc=0.0409, homo_pred=0.0409 -- the model's worst case). These rows are the first genuinely heterozygous real-read diploid test.

| founderA | founderB | depth_per_hap | n_placed | n_unplaced | self_cov_A_pct | self_cov_B_pct | het_frac | n_sites | pair_acc | hap_acc | homo_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B73 | CML247 | 250000 | 370047 | 6468 | 67.6112 | 62.4809 | 1.0000 | 314880 | 0.7334 | 0.8100 | 0.0015 |
| B73 | CML247 | 500000 | 739993 | 12777 | 67.6742 | 62.4599 | 1.0000 | 625664 | 0.6321 | 0.7373 | 0.0028 |
| B73 | CML247 | 1000000 | 1481101 | 25745 | 67.7243 | 62.3805 | 1.0000 | 1231360 | 0.5400 | 0.6762 | 0.0040 |
| B73 | Oh43 | 250000 | 373259 | 5389 | 68.7465 | 69.0374 | 1.0000 | 342016 | 0.7630 | 0.8647 | 0.0023 |
| B73 | Oh43 | 500000 | 746882 | 10663 | 68.7868 | 69.0984 | 1.0000 | 679424 | 0.6735 | 0.8135 | 0.0022 |
| B73 | Oh43 | 1000000 | 1494821 | 20792 | 68.7712 | 69.1664 | 1.0000 | 1336832 | 0.5972 | 0.7722 | 0.0027 |
| CML247 | CML277 | 250000 | 357652 | 9748 | 68.0546 | 67.4216 | 1.0000 | 288768 | 0.8077 | 0.8584 | 0.0023 |
| CML247 | CML277 | 500000 | 716262 | 19322 | 67.9750 | 67.4763 | 1.0000 | 573440 | 0.7267 | 0.7983 | 0.0049 |
| CML247 | CML277 | 1000000 | 1433171 | 38831 | 67.9396 | 67.4712 | 1.0000 | 1127424 | 0.6267 | 0.7244 | 0.0077 |

## Reference points

- `diploid-sim512-h3` (held-out simulated test split): pair_acc=0.6186
- `diploid-sim-on-ropebwt-oh43-k24` (homozygous real Oh43): pair_acc=0.0409, homo_pred=0.0409

