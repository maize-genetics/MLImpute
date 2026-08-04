### GRITS Testing Suite (Bella)

## Software

# Conda Enviorments

https://biohpc.cornell.edu/lab/doc/Software_Installation_exercises1.html

# Impute5

Documentation:

https://www.dropbox.com/scl/fo/ukwimchnvp3utikrc3hdo/AKqYvE6-9C5kLpKDSfhR8xQ?rlkey=n2zty39bdst5j5tycd0sf89ee&e=1&dl=0

impute5_v1.2.0 -> doc -> IMPUTE5.pdf

Software:

biohpc has most updated version

export PATH=/programs/impute5_v1.2.0:$PATH

Tutorial Tips:

- add --buffer-region parameter with 20:900000-4100000 (not listed as required but always gives error if missing)

# Beagle5.5

Documentation:

https://faculty.washington.edu/browning/beagle/beagle_5.5_17Dec24.pdf

Software:

conda create -n imputation python=3.6 anaconda
conda activate imputation
conda install -c bioconda eagle
conda install -c bioconda beagle
conda install -c r r-base
conda install -c bioconda bcftools
conda install -c conda-forge r-data.table
conda install -c conda-forge r-sm

Tutorial: 

https://www.protocols.io/run/genotype-imputation-workflow-v3-0-xbgfijw


Tutorial Tips:
- /workdir/irk9/test_imputation_models/beagle/chr20_b37_beagle_ex/target → java -Xmx16g -jar /programs/beagle41/beagle41.jar gt=target_chr20_b37_02.vcf.gz ref=../ref/chr20.1kg.phase3.v5a.rm_target_samples_02.vcf.gz map=../maps/beagle_chr20_b37_nochr.map out=../output/beagle_chr20_b37_08.vcf.gz
- I took the reference genome vcf for chr20 of b37 and removed target_samples_02 to make chr20.1kg.phase3.v5a.rm_target_samples_02.vcf.gz and then made target_chr20_b37_02.vcf.gz by only keeping the samples from the reference genome that are listed in target_samples_02.txt
- With beagle conda env only need to write beagle instead of java -Xmx16g -jar /programs/beagle41/beagle41.jar


# Minimac4

Documentation:

https://github.com/statgen/minimac4

Software:

cget install --prefix <install_prefix> statgen/Minimac4

Tutorial Tips:

- biohpc minimac3 is updated (use Minimac3 keyword), but minimac4 is not


# Glimpse

Documentation:

https://odelaneau.github.io/GLIMPSE/docs/documentation


Software:

biohpc has most updated version

export PATH=/programs/glimpse_v2.0.0:$PATH

Tutorial:

https://odelaneau.github.io/GLIMPSE/docs/tutorials/getting_started/

Tutorial Tips:

- Look through different sh files in the github tutorial folder. The website tutorial version only has most (but not all) of the code (https://github.com/odelaneau/GLIMPSE/tree/master/tutorial)
- If using biohpc, remember to remove ./bin/ from tutorial scripts

# BioKotlin Tools

https://github.com/maize-genetics/biokotlin-tools

# Other Software

- Bcftools: https://samtools.github.io/bcftools/bcftools.html 
- Samtools: https://www.htslib.org/doc/ 
- BWA: https://bio-bwa.sourceforge.net/ 
- PHG: https://phg.maizegenetics.net/


## Files

# Coordinate Fastas

Maize: /workdir/irk9/data/phg-maize/trial/B73.fa
Cassava: /workdir/smm477/phg-cassava/ref/Mesculenta_671_v8.0.fa


# Reference Panels


Test 1 and 2:

Maize

Impute: /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz
Minimac: /workdir/irk9/data/phg-maize/truth-vcfs/maize_pangenome_snps.msav 
Beagle: /workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr
Grits:  /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz

Cassava

Impute: /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz
Minimac: /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.msav 
Beagle: /workdir/irk9/data/phg-cassava/truth-vcfs/beagle_ref/ref_split
Grits: /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.vcf.gz


Test 3:


Maize

Impute: /workdir/irk9/data/phg-maize/test3/maize_pangenome_snps_cleaned_phased_AN_AC_nomissing.vcf.gz
Minimac: /workdir/irk9/data/phg-maize/test3/maize_test3_ref.msav
Beagle: /workdir/irk9/data/phg-maize/test3/ref_panel/beagle_ref
Grits: /workdir/irk9/data/phg-maize/test3/ref_panel/maize_sim_merged_biallelic_chr.gvcf

Cassava

Impute: /workdir/irk9/data/phg-cassava/test3/ref_panel/cassava_sim_merged_biallelic_diploid_header_AN_AC_nomissing_rename_chr.vcf.gz
Minimac: /workdir/irk9/data/phg-cassava/test3/ref_panel/cassava_test3_ref.msav
Beagle: /workdir/irk9/data/phg-cassava/test3/ref_panel/beagle_ref
Grits: /workdir/ahb232/bella_seq_sim/cassava/merged_vcfs/cassava_sim_merged_biallelic.gvcf


# Splines

Test 1 and 2:

Maize: /workdir/smm477/phg-maize/splines
Cassava: /workdir/smm477/phg-cassava/splines

Test 3:

Maize:  /workdir/ahb232/bella_seq_sim/maize/splines
Cassava: /workdir/ahb232/bella_seq_sim/cassava/splines

# Index

Test 1 and 2:

Maize: /workdir/smm477/phg-maize/ropebwt-index-ML/ropebwt_index.fmd
Cassava: /workdir/smm477/phg-cassava/ropebwt-index-ML/ropebwt_index.fmd

Test 3:

Maize: /workdir/ahb232/bella_seq_sim/maize/ropebwt_index/phgIndex.fmd
Cassava: /workdir/ahb232/bella_seq_sim/cassava/ropebwt_index/cassava_sim.fmd


# Chr Ends

Maize: /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt

Cassava: /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt

# Maps (That didn't work)

/workdir/irk9/data/maps/

## Scripts



## Genomics 101 Resources


Pre-GWAS-Post Key Concept review: https://cloufield.github.io/GWASTutorial/Imputation/ 
GWAS: https://www.youtube.com/watch?v=sOP8WacfBM8
Linkage Disequilibrium: https://www.youtube.com/watch?v=C3MYoasLSHQ
Genomic Selection: https://acsess.onlinelibrary.wiley.com/doi/10.1002/tpg2.70053 
Mapping Short Read Fastq to Reference Fasta: https://bio-bwa.sourceforge.net/ 
Sorghum PHG: https://acsess.onlinelibrary.wiley.com/doi/10.1002/tpg2.20009
Evolutionary Rescue of Sorghum: https://www.science.org/doi/10.1126/sciadv.abj4633#core-R24-1 
Chain File: https://genome.ucsc.edu/goldenpath/help/chain.html 



