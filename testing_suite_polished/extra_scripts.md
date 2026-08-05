# Extra scripts

## Disclaimer

This is a collection of commands that can be used to clean reference panels or other purposes. It isn't a normal script that can be ran all at once.

## Lift Coordinates

### MAF file
Use anchorwave and then the proali command with R = Q = 1 where old_version as ref and new_version as query with -i as ref gff file on new_version to produce a maf file

### Maf to chain
Use Nf-core https://biohpc.cornell.edu/lab/userguide.aspx?a=software&i=1311#c 
Conda install last https://anaconda.org/channels/bioconda/packages/last/overview
Download maf-covert https://nf-co.re/modules/last_mafconvert/
Then use command maf-convert psl my-alignments.maf > my-alignments.psl with documentation at https://gitlab.com/mcfrith/last/-/blob/main/doc/maf-convert.rst?ref_type=heads 

### Chain file + old_version bed file ⇒ new_version bed file
Use cross map https://biohpc.cornell.edu/lab/userguide.aspx?a=software&i=286#c


## Alignment tools

### Anchor wave

https://github.com/baoxingsong/AnchorWave/blob/master/FAQ.md 


Example from github:
anchorwave gff2seq -r Zea_mays.AGPv4.dna.toplevel.fa -i Zea_mays.AGPv4.34.gff3 -o cds.fa
minimap2 -x splice -t 10 -k 12 -a -p 0.4 -N 20 Sorghum_bicolor.Sorghum_bicolor_NCBIv3.dna.toplevel.fa cds.fa > cds.sam
minimap2 -x splice -t 10 -k 12 -a -p 0.4 -N 20 Zea_mays.AGPv4.dna.toplevel.fa cds.fa > ref.sam
anchorwave proali -i Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3 -as cds.fa -r /workdir/irk9/data/phg-maize/test/B73.fa -a cds.sam -ar ref.sam -s B73_RefGen_v3.fa -n anchors -R 1 -Q 1 -o B73_alignment.maf -f B73_alignment.f.maf


## Minimac ref vcf to msav

minimac4 --compress-reference ref.vcf.gz > ref.msav


## Glipmse test (didn't work/in progress)

Error: not enough data in reference panel. Github repo creator suggested having 50 samples in panel.

Code below was used, but not tested that well

### ref panel prep

bcftools norm -m -any maize_pangenome_snps.vcf.gz --threads 4 -Ou | bcftools view -m 2 -M 2 -v snps --threads 4 -Ob -o ref.bcf
bcftools index ref.bcf
bcftools view -G -Oz -o ref.vcf.gz ref.bcf
bcftools index ref.vcf.gz
GLIMPSE2_chunk --input ref.vcf.gz --region chr1 --output chunks.chr1.txt --sequential
bcftools +fill-tags ref.bcf -Ob -o ref.fixed.bcf -- -t AC,AN
bcftools view   -e 'F_MISSING>0'   -Ob   -o ref.no_missing.bcf   ref.bcf
 bcftools index ref.no_missing.bcf 
bcftools +fill-tags ref.no_missing.bcf -Ob -o ref.acan.bcf  -- -t AC,AN
bcftools index ref.acan.bcf
REF=ref.acan.bcf
while IFS="" read -r LINE || [ -n "$LINE" ];
do
  printf -v ID "%02d" $(echo $LINE | cut -d" " -f1)
  IRG=$(echo $LINE | cut -d" " -f3)
  ORG=$(echo $LINE | cut -d" " -f4)


  GLIMPSE2_split_reference --reference ${REF} --input-region ${IRG} --output-region ${ORG} --output /workdir/irk9/data/phg-maize/test_glimpse/ref_panel/split_ref/
done < chunks.chr1.txt


### Split reference

GLIMPSE2_split_reference --input-region chr1:1-308300047 --output-region chr1:1-308300047 --output /workdir/irk9/data/phg-maize/test_glimpse/ref_panel/split_ref_whole/ --reference ref.acan.bcf

### Phase step

GLIMPSE2_phase --bam-file /workdir/irk9/data/phg-maize/test_glimpse/target/B97_sorted.bam --reference /workdir/irk9/data/phg-maize/test_glimpse/ref_panel/split_ref_whole/_chr1_1_308300047.bin --output /workdir/irk9/data/phg-maize/test_glimpse/impute/imputed.bcf


GLIMPSE2_phase --bam-file /workdir/irk9/data/phg-maize/test_glimpse/2x/B97_sorted.bam --reference /workdir/irk9/data/phg-maize/test_glimpse/ref_panel/split_ref_whole/_chr1_1_308300047.bin --output /workdir/irk9/data/phg-maize/test_glimpse/2x/imputed.bcf


## Beagle Clean Ref Panel

### clean ref panel

bcftools view maize_pangenome_snps.vcf.gz -e 'INFO/AC<3 | INFO/AN-INFO/AC<3' -Ou | \
bcftools norm -m -any -Ou | \
bcftools view -v snps,indels -Ou | \
bcftools norm -f /workdir/irk9/data/phg-maize/trial/B73.fa -d none -Ou | \
bcftools view -m 2 -M 2 -Ou | \
bcftools view -g ^miss -Oz -o ref.vcf.gz
bcftools index ref.vcf.gz

### phase reference panel

bcftools +fixploidy \
    ref.vcf.gz -Ov | \
bcftools view -e 'GT[*]="mis"' -Ov |
awk 'BEGIN { OFS="\t" }
  /^#/ {
    print
    next
  }
  {
    for (i = 10; i <= NF; i++) {
      split($i, fields, ":")
      gsub("/", "|", fields[1])
      $i = fields[1]
      for (j = 2; j <= length(fields); j++) {
        $i = $i ":" fields[j]
      }
    }
    print
  }' | bgzip -c > ref_phased.vcf.gz


### remove duplicate ids

bcftools query -f '%ID\n' ref.vcf.gz | grep -v '^\.$' | sort | uniq -d | wc -l

** maize didn't have duplicate ids

### AF cleaning

bcftools +fill-tags ref.vcf.gz -Oz -o ref_AF.vcf.gz -- -t AF
bcftools index ref_AF.vcf.gz
echo -e 'CHR\tSNP\tREF\tALT\tAF' > ref_imputation.frq
bcftools query \
  -f '%CHROM\t%CHROM\_%POS\_%REF\_%ALT\t%REF\t%ALT\t%INFO/AF\n' \
  ref_AF.vcf.gz \
  >> ref_imputation.frq


### split Chr

VCF="ref_AF.vcf.gz"


while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="ref_AF_${CHR}.vcf.gz"


   bcftools view \
       -r "${CHR}" \
       "${VCF}" \
       -Oz \
       -o "${OUT_VCF}"
  bcftools index "${OUT_VCF}"


done < chromosomes.txt



### collect sample ids

bcftools query -l ref_AF_chr1.vcf.gz > ref_AF_sample_IDs.txt

I don't think this step was ever used for anything

### filter out non-chr data

for CHR in chr{1..10}; do
   INVCF="ref_AF_${CHR}.vcf.gz"
   OUTVCF="ref_AF_${CHR}_chrfiltered.vcf.gz"


   echo "Filtering ${INVCF}"


   bcftools view \
       -t "${CHR}" \
       "${INVCF}" \
       -Oz \
       -o "${OUTVCF}"


   bcftools index -t "${OUTVCF}"
Done


## How to make an answer key

### gvcf to vcf

bcftools view -m 3 --trim-alt-alleles out.g.vcf.gz -Oz -o maize_test2.vcf.gz

### Only keep AN, AC, GT info

bcftools annotate --remove "^INFO/AC,INFO/AN" maize_test2.vcf.gz -Oz -o maize_test2_AN_AC.vcf.gz
bcftools index maize_test2_AN_AC.vcf.gz
bcftools annotate --remove "^FORMAT/GT" maize_test2_AN_AC.vcf.gz -Oz -o maize_test2_AN_AC_GT.vcf.gz


### Phase

bcftools +fixploidy  maize_test2_AN_AC_GT.vcf.gz -Oz -o  maize_test2_AN_AC_GT_phased.vcf.gz

### Merge

bcftools merge --missing-to-ref /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz maize_test2_AN_AC_GT_phased.vcf.gz -Oz -o maize_test2_merged_fixed.vcf.gz --threads 10

### Filter (SNPS only)

bcftools view -v input.vcf -Oz -o output_snps_only.vcf.gz