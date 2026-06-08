# How to generate bed files from ML numpy predictions

call from the main grits dir:

```bash
python -m src.python.bed_io.inference_to_bed
    --inference-dir /path/to/numpy/pred/dir 
    --ps4g-dir /path/to/corresponding/ps4g/dir 
    --save-dir /save/bed/files/here 
    --contiguous 
    --fai /path/to/ref/fai
```

* numpy predictions are only for the positions included in the corresponding ps4g file
* the --contiguous option will extend out the positions of these predictions so that every reference coordinates has a prediction
  * The "unpredicted" chunks are split in half between their adjacent predictions

This script generates one bed file per chromosome per sample
* In order to run sample bed-to-vcf, the chromosomes for each sample should be added to one bed file
  * Otherwise, it create a new sample entry in the vcf for each chromosome
* Run this command to concatenate chromosome bed files for each sample (switch Chromosome to whatever format contigs are named)
  ```bash
    mkdir -p all_chr && for sample in $(ls *_Chromosome*_imputed.contiguous.bed | sed 's/_Chromosome.*//g' | sort -u); do { head -1 $(ls ${sample}_Chromosome*_imputed.contiguous.bed | head -1); tail -n +2 -q ${sample}_Chromosome*_imputed.contiguous.bed | sort -k1,1 -k2,2n; } > all_chr/${sample}.bed; done
  ```