# GRITS Testing Suite (Bella)

## Table of Content

- software
- important files
- scripts
- genomics 101 resources

## Software

### Bella's google doc notes with basically the same information, but the markdown version is more readable

https://docs.google.com/document/d/1K0Fjc6RXTB44s6OFpClQBEPgcV0TKWHA7PxwCYbYoGw/edit?usp=sharing

I added it just in case I forgot to add something important to my readme.md

### Conda Enviorments

https://biohpc.cornell.edu/lab/doc/Software_Installation_exercises1.html

### Impute5

Documentation:

https://www.dropbox.com/scl/fo/ukwimchnvp3utikrc3hdo/AKqYvE6-9C5kLpKDSfhR8xQ?rlkey=n2zty39bdst5j5tycd0sf89ee&e=1&dl=0

impute5_v1.2.0 -> doc -> IMPUTE5.pdf

Software:

biohpc has most updated version

export PATH=/programs/impute5_v1.2.0:$PATH

Tutorial Tips:

- add --buffer-region parameter with 20:900000-4100000 (not listed as required but always gives error if missing)

### Beagle5.5

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


### Minimac4

Documentation:

https://github.com/statgen/minimac4

Software:

cget install --prefix <install_prefix> statgen/Minimac4

Tutorial Tips:

- biohpc minimac3 is updated (use Minimac3 keyword), but minimac4 is not


### Glimpse

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

### BioKotlin Tools

https://github.com/maize-genetics/biokotlin-tools

### Other Software

- Bcftools: https://samtools.github.io/bcftools/bcftools.html 
- Samtools: https://www.htslib.org/doc/ 
- BWA: https://bio-bwa.sourceforge.net/ 
- PHG: https://phg.maizegenetics.net/

### How to run tutorials for each

/workdir/irk9/test_imputation_models

All of my inital tests to get the models working are in here


## Files

### /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/important_files has all the same files too

Below is all the original paths in case I missed a file somewhere you can check the adjacent dir

### Coordinate Fastas

Maize: /workdir/irk9/data/phg-maize/trial/B73.fa

Cassava: /workdir/smm477/phg-cassava/ref/Mesculenta_671_v8.0.fa


### Reference Panels


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

Impute: /workdir/irk9/data/phg-maize/test3/ref_panel/maize_pangenome_snps_cleaned_phased_AN_AC_nomissing.vcf.gz

Minimac: /workdir/irk9/data/phg-maize/test3/ref_panel/maize_test3_ref.msav

Beagle: /workdir/irk9/data/phg-maize/test3/ref_panel/beagle_ref


Grits: /workdir/irk9/data/phg-maize/test3/ref_panel/maize_sim_merged_biallelic_chr.gvcf


Cassava

Impute: /workdir/irk9/data/phg-cassava/test3/ref_panel/cassava_sim_merged_biallelic_diploid_header_AN_AC_nomissing_rename_chr.vcf.gz

Minimac: /workdir/irk9/data/phg-cassava/test3/ref_panel/cassava_test3_ref.msav

Beagle: /workdir/irk9/data/phg-cassava/test3/ref_panel/beagle_ref

Grits: /workdir/ahb232/bella_seq_sim/cassava/merged_vcfs/cassava_sim_merged_biallelic.gvcf



### Splines

Test 1 and 2:

Maize: /workdir/smm477/phg-maize/splines

Cassava: /workdir/smm477/phg-cassava/splines

Test 3:

Maize:  /workdir/ahb232/bella_seq_sim/maize/splines

Cassava: /workdir/ahb232/bella_seq_sim/cassava/splines

### Index

Test 1 and 2:

Maize: /workdir/smm477/phg-maize/ropebwt-index-ML/ropebwt_index.fmd

Cassava: /workdir/smm477/phg-cassava/ropebwt-index-ML/ropebwt_index.fmd

Test 3:

Maize: /workdir/ahb232/bella_seq_sim/maize/ropebwt_index/phgIndex.fmd

Cassava: /workdir/ahb232/bella_seq_sim/cassava/ropebwt_index/cassava_sim.fmd


### Chr Ends

Maize: /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt

Cassava: /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt

### Answer keys for test2:

Maize: /workdir/shared_files/maize_test2_answer_key

Cassava: /workdir/shared_files/cassava_test2_answer_key

### GRITS-enc-dec ckpt



### Maps (That didn't work)

/workdir/irk9/data/maps/

## Scripts

### Disclaimer

I polished my scripts and made sure that any confusing parts or inconsistencies are fixed in the testing_suite_bella folder, but I didn't have enough time to throughly test it so you may run into random little errors. The content used to impute, clean, and evalute is correct though and you can review my raw scripts in bellas_scripts. Those work well.

Some of the software paths are hardcoded, but the software part of the readme explains how to download it yourself. grits.sh and beagle.sh don't have hardcoded conda env paths.

### Test scripts / learn how to use them

/workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test

I ran through all my scripts in this dir to try and quickly check them before I left.

### fq_to_vcf.sh

Purpose:

Turn fastq reads into a target vcf using coordinate reference.

Usage: 

./fq_to_vcf.sh --reads DIR --coordinate_ref FASTA --out DIR options

Required:
  --reads DIR          Directory containing .fastq.gz files
  --coordinate_ref FA  Reference FASTA (must be indexed with samtools faidx and bwa index)
  --out DIR            Output directory

Optional:
  --list NAMES         Comma-separated sample names to process (no .fastq.gz extension)
                       When absent, all .fastq.gz files in --reads are processed
  --threads N          Number of threads (default: 2)
  --help               Show this message and exit

Examples:
  Process all samples in a directory:
  $(basename "$0") --reads /data/reads --coordinate_ref /data/ref.fa --out /data/output --threads 15

  Process specific samples only:
  $(basename "$0") --reads /data/reads --coordinate_ref /data/ref.fa --out /data/output --list Mo18W,Ms71,NC350

### clean.sh

Purpose:

Clean target vcf files for impute and beagle

Usage:
./clean.sh --step STEP --target-vcf-dir DIR --ref-panel-vcf FILE --out-dir DIR options

Required arguments:
  --target-vcf-dir DIR      Directory containing target VCF files
  --ref-panel-vcf FILE      Reference panel VCF file
  --out-dir DIR             Output directory

Optional arguments:
  --step STEP               Step to run: clean, validate_cleaning, convert_bcf, make_xcf, make_chunks, all
                            Default: all
  --rename-chr FILE         Chromosome rename file (required for cassava, skip for maize)
  --map-dir DIR             Directory containing genetic maps
  --threads INT             Number of threads
                            Default: 5
  -h, --help                Show this help message


### impute.sh

read usage function in file

### minimac.sh

read usage function in file

### beagle.sh

read usage function in file

### grits.sh

read usage function in file

### accuracy.py

Purpose:

Sarah's script that evaluates snp accuracy, rare allele accuracy, and R^2.

Usage example: 

python accuracy.py --truth /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --imputed /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/vcf/renamed.vcf -s Oh7B -r chr1 --partial-credit

### run_accuracy.py


### eval_results.ipynb

Purpose:

Script that was used to create the graphs I used for my talk.

### fix_bed_boundaries.py

Purpose:

Bandaid fix for start and end chromosomes in grits output bed file, because impute.py needs to be updated.

It is a helper function and you don't need to run it directly. It is used by grits.sh

### calc_major_af.py

Purpose:

Helps you calcuate the default rate or major allele frequency within a vcf file.

Usage example:

python calc_major_af.py input.vcf

### sum_results.py

Purpose:

If you have a results.tsv file with evaluation data from the same test and species but different samples and you want to evaluate them as a whole and not per sample, use this cause it will average the accuracy and R^2 values.

Usage:

python -i /dir/results.tsv -o /dir/sum_results.tsv

input tsv is usually from the run_accuracy.sh script. Its that results.tsv file

### extra_scripts.md

Learn how to clean the reference panels and about any other extra code I might have use all labeled an organized.

Its more of commands to run to complete said goal versus on script that will do it.

### maps directory

Files I used to lift over coordinates, reformat information in bed files, and condense map information.

We decided not to use the maps I created because the markers in the published map were not the ones we were looking for.

I kept the map scripts, but they are not polished.


## Genomics 101 Resources


Pre-GWAS-Post Key Concept review: https://cloufield.github.io/GWASTutorial/Imputation/ 

GWAS: https://www.youtube.com/watch?v=sOP8WacfBM8

Linkage Disequilibrium: https://www.youtube.com/watch?v=C3MYoasLSHQ

Genomic Selection: https://acsess.onlinelibrary.wiley.com/doi/10.1002/tpg2.70053 

Mapping Short Read Fastq to Reference Fasta: https://bio-bwa.sourceforge.net/ 

Sorghum PHG: https://acsess.onlinelibrary.wiley.com/doi/10.1002/tpg2.20009

Evolutionary Rescue of Sorghum: https://www.science.org/doi/10.1126/sciadv.abj4633#core-R24-1 

Chain File: https://genome.ucsc.edu/goldenpath/help/chain.html 

