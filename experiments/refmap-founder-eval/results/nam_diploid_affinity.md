# Synthetic-diploid baseline (combined real founder reads, affinity diploid CRF)

Each row combines two founders' real WGS read files (`WGSReads/`) -- a `head`-based subsample of `depth_per_hap` reads from each, concatenated -- run through the same `ropebwt3 refmap` whole-read placement recipe as `nam_baseline.py` and windowed to K=24 (`crf/ropebwt_npy_to_matrix.py --window-size=512 --target-num-parents=24`). Because the two read sets are combined wholesale (no recombination), truth is exactly `(founderA, founderB)` at every site -- the label columns are overwritten with this known-by-construction pair, and `het_frac` is asserted to be 100%.

Scored with the **affinity-conditioned** (`founder_affinity=True`, `checkpoints/diploid-affinity-sim512-h3`) diploid GRITS-CRF, fed a genome-wide `_founder_affinity` vector computed from this run's own data (label-free) as `ext_emb` -- crf/eval.py's plain `evaluate_diploid` never threads `ext_emb` for diploid mode, so this uses a local `evaluate_diploid_with_affinity` (see `run_diploid_eval`/`evaluate_diploid_with_affinity` in this file). These rows reuse the identical cached reads/refmap/windowed matrices as `results/nam_diploid.md` (the original plain-model run) -- only the final scoring step differs, so any difference in `pair_acc` between the two reports is attributable to the affinity conditioning alone.

| founderA | founderB | depth_per_hap | n_placed | n_unplaced | self_cov_A_pct | self_cov_B_pct | het_frac | n_sites | pair_acc | hap_acc | homo_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B73 | CML247 | 250000 | 370047 | 6468 | 67.6112 | 62.4809 | 1.0000 | 314880 | 0.9534 | 0.9707 | 0.0017 |
| B73 | CML247 | 500000 | 739993 | 12777 | 67.6742 | 62.4599 | 1.0000 | 625664 | 0.9191 | 0.9480 | 0.0030 |
| B73 | CML247 | 1000000 | 1481101 | 25745 | 67.7243 | 62.3805 | 1.0000 | 1231360 | 0.8705 | 0.9144 | 0.0035 |
| B73 | Oh43 | 250000 | 373259 | 5389 | 68.7465 | 69.0374 | 1.0000 | 342016 | 0.9630 | 0.9806 | 0.0014 |
| B73 | Oh43 | 500000 | 746882 | 10663 | 68.7868 | 69.0984 | 1.0000 | 679424 | 0.9447 | 0.9711 | 0.0018 |
| B73 | Oh43 | 1000000 | 1494821 | 20792 | 68.7712 | 69.1664 | 1.0000 | 1336832 | 0.9208 | 0.9581 | 0.0035 |
| CML247 | CML277 | 250000 | 357652 | 9748 | 68.0546 | 67.4216 | 1.0000 | 288768 | 0.9572 | 0.9694 | 0.0025 |
| CML247 | CML277 | 500000 | 716262 | 19322 | 67.9750 | 67.4763 | 1.0000 | 573440 | 0.9158 | 0.9383 | 0.0046 |
| CML247 | CML277 | 1000000 | 1433171 | 38831 | 67.9396 | 67.4712 | 1.0000 | 1127424 | 0.8510 | 0.8921 | 0.0079 |

## Reference points

- `diploid-sim512-h3` (held-out simulated test split, plain): pair_acc=0.6186
- `diploid-affinity-sim512-h3` (held-out simulated test split, affinity): pair_acc=0.6179
- `diploid-sim-on-ropebwt-oh43-k24` (homozygous real Oh43, plain): pair_acc=0.0409, homo_pred=0.0409
- See `results/nam_diploid.md` for the plain-model numbers on these same pairs/depths.

