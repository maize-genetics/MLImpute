# Synthetic-diploid Tripsacum baseline (simulated reads, plain diploid CRF)

Each row simulates reads directly from two assembly FASTAs (`wgsim -e 0.001 -r 0 -R 0 -1 100 -2 100`, R1 only kept), combines them into one FASTQ with no recombination (truth = `(assemblyA, assemblyB)` at every site by construction), runs the combined reads through `ropebwt3 refmap` against the Tripsacum 18-sample pangenome (`tripsacumChrIndex.{fmd,lift}`, `--ref-prefix=Td-FL-9056069-6-REFERENCE-PanAnd-2.0a_`), windows at the true panel size K=18 (`crf/ropebwt_npy_to_matrix.py --window-size=512`), then pads to K=24 with 6 permanently-zero founder columns to match `checkpoints/diploid-sim512-h3`'s fixed 24-founder architecture (verified via direct checkpoint inspection -- see script docstring). Scored with the plain (non-affinity; `founder_affinity=False`) diploid GRITS-CRF via `crf/eval.py::evaluate_diploid`.

Pairs were chosen by measured genetic relatedness (`mash dist`, `tripsacum/relatedness/all_pairs_dist.tsv` -- no accession metadata exists, and no distance tool was previously installed): the starter pair C009-T009 x C011-T007 (dist=0.009366), then the closest-distance matches among all other unordered pairs of the 14 non-reference assemblies (0.009406, 0.009206, and the closest fully-independent-member pair at 0.006910).

| assemblyA | assemblyB | depth_per_hap | n_placed | n_unplaced | self_cov_A_pct | self_cov_B_pct | het_frac | n_sites | pair_acc | hap_acc | homo_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C009-T009 | C011-T007 | 250000 | 330142 | 21885 | 76.9089 | 74.5079 | 1.0000 | 316416 | 0.7632 | 0.8711 | 0.0001 |
| C009-T009 | C027-T007 | 250000 | 330869 | 22018 | 76.8350 | 74.9918 | 1.0000 | 317440 | 0.8811 | 0.9321 | 0.0001 |
| C009-T009 | C050-T007 | 250000 | 332908 | 21507 | 76.4366 | 74.7133 | 1.0000 | 318464 | 0.8793 | 0.9394 | 0.0003 |
| C076-T198 | C081-T199 | 250000 | 351785 | 24127 | 82.3696 | 82.1901 | 1.0000 | 342528 | 0.9852 | 0.9901 | 0.0016 |

