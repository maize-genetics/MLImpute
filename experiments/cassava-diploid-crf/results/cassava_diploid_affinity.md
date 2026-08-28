# Synthetic-diploid cassava baseline (simulated reads, affinity diploid CRF)

Each row simulates reads directly from two assembly FASTAs (`wgsim -e 0.001 -r 0 -R 0 -1 100 -2 100`, R1 only kept), combines them into one FASTQ with no recombination (truth = `(assemblyA, assemblyB)` at every site by construction), runs the combined reads through `ropebwt3 refmap` against the cassava 80-founder pangenome (`cassavaChrIndex.{fmd,lift}`, `--ref-prefix=Mesculenta-671-v8-0_`), windows at the true panel size K=80 (`crf/ropebwt_npy_to_matrix.py --window-size=512`), then SELECTS DOWN to K=24 (true pair force-included + the 22 next most genome-wide-covered founders) to match the checkpoint's fixed 24-founder architecture -- the reverse of Tripsacum's 18->24 pad-up, since cassava's indexed panel (80) is larger than the model. Scored with the **affinity-conditioned** (`founder_affinity=True`, `checkpoints/diploid-affinity-sim512-h3`) diploid GRITS-CRF, fed a genome-wide `_founder_affinity` vector computed from this run's own data.

`kind=within` rows pair the two haplotypes of the SAME accession (a genuinely real heterozygous individual, unique to cassava's hap-resolved assemblies among our datasets); `kind=cross` rows pair different accessions chosen by mash distance (`cassava/relatedness/all_pairs_dist.tsv`).

| kind | assemblyA | assemblyB | depth_per_hap | n_placed | n_unplaced | self_cov_A_pct | self_cov_B_pct | cov_rank_A | cov_rank_B | het_frac | n_sites | pair_acc | hap_acc | homo_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross | DSC493-12-1-hap2-v2 | COL40-DSC118-hap2-v2 | 250000 | 357501 | 4822 | 88.4459 | 88.4164 | 0 | 1 | 1.0000 | 342528 | 0.9680 | 0.9791 | 0.0003 |
| cross | TMEB693-hap1-v2 | TMe-2497-G5-hap2-v2 | 250000 | 344232 | 4675 | 89.8392 | 90.1563 | 0 | 1 | 1.0000 | 329216 | 0.9038 | 0.9232 | 0.0000 |
| cross | TMEB693-hap2-v2 | CR63-hap1-v2 | 250000 | 343815 | 6041 | 84.4848 | 84.6905 | 0 | 1 | 1.0000 | 328704 | 0.7804 | 0.8436 | 0.0000 |
| cross | TMEB693-hap2-v2 | TMe-2497-G5-hap1-v2 | 250000 | 345125 | 4230 | 90.1799 | 90.3162 | 0 | 1 | 1.0000 | 328704 | 0.9217 | 0.9404 | 0.0000 |
| within | BGM-2098-hap1-v2 | BGM-2098-hap2-v2 | 250000 | 366702 | 6872 | 68.9572 | 69.5765 | 0 | 1 | 1.0000 | 353280 | 0.8423 | 0.8792 | 0.0052 |
| within | COL386-hap1-v3 | COL386-hap2-v3 | 250000 | 365468 | 4517 | 70.3713 | 71.7095 | 0 | 1 | 1.0000 | 351744 | 0.8965 | 0.9214 | 0.0104 |
| within | IITA-TMS-IBA000070-hap1-v2 | IITA-TMS-IBA000070-hap2-v2 | 250000 | 328771 | 5370 | 69.1016 | 70.5947 | 0 | 1 | 1.0000 | 315392 | 0.6883 | 0.7529 | 0.0019 |
| within | TME204-hap1-v2 | TME204-hap2-v2 | 250000 | 347525 | 5129 | 67.2133 | 68.4960 | 0 | 1 | 1.0000 | 333824 | 0.5800 | 0.6802 | 0.0006 |
| within | TMe-3055-G9-hap1-v2 | TMe-3055-G9-hap2-v2 | 250000 | 359400 | 7849 | 67.3903 | 68.1687 | 0 | 1 | 1.0000 | 345088 | 0.6344 | 0.7139 | 0.0020 |

