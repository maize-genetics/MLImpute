in_bcf="ref.acan.bcf"
samples_file="./ref_50/empty_samples.txt"
dummy_vcf="./ref_50/empty_samples.vcf.gz"
out_bcf="./ref_50/input.plus_26_empty_samples.bcf"


# Build the sample columns for the VCF header
sample_header=$(awk '{printf "\t%s", $1}' "$samples_file")

# Build one missing GT field per empty sample
missing_gts=$(awk 'BEGIN { first=1 } { printf "%s.", first ? "" : "\t"; first=0 }' "$samples_file")

# Create a dummy multi-sample VCF with the same sites
{
  bcftools view -h "$in_bcf" | awk -v sample_header="$sample_header" '
    BEGIN { hasGT=0 }
    /^##FORMAT=<ID=GT,/ { hasGT=1 }
    /^#CHROM/ {
      if (!hasGT) {
        print "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
      }
      print "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT" sample_header
      next
    }
    { print }
  '

  bcftools query -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t.\tPASS\t.\tGT\n' "$in_bcf" \
    | awk -v missing_gts="$missing_gts" '
      {
        print $0 "\t" missing_gts
      }
    '
} | bgzip -c > "$dummy_vcf"



indir="/workdir/irk9/data/maps/split_by_pos"
outdir="/workdir/irk9/data/maps/split_by_pos_cleaned_no_neg"

mkdir -p "$outdir"

for file in "$indir"/*.map; do
    base=$(basename "$file" .map)
    outfile="$outdir/${base}_cleaned.map"

    awk '
        BEGIN {
            prev_cm = ""
        }

        # Keep header if present
        NR == 1 && ($1 ~ /pos|POS/ || $2 ~ /chr|CHR/ || $3 ~ /cM|CM/) {
            print
            next
        }

        {
            pos = $1
            chr = $2
            cm  = $3

            if (cm < 0) {
                next
            }

            if (prev_cm == "" || cm >= prev_cm && cm >= 0) {
                print
                prev_cm = cm
            }
        }
    ' "$file" > "$outfile"

    echo "Wrote $outfile"
done


for f in /workdir/irk9/data/maps/split_by_pos_cleaned_no_neg/*.map
do
    awk 'NR > 1 && $3 < 0 { print FILENAME, "line", NR, $0 }' "$f"
done



indir="/workdir/irk9/data/maps/split_by_pos_cleaned_no_neg"
outfile="/workdir/irk9/data/maps/split_by_pos_cleaned_no_neg/cassava.map"

first=1

> "$outfile"

for file in "$indir"/*.map
do
    [ -e "$file" ] || continue

    if [ "$first" -eq 1 ]; then
        cat "$file" >> "$outfile"
        first=0
    else
        tail -n +2 "$file" >> "$outfile"
    fi
done

echo "Wrote $outfile"


tr -d ' \n' < file.txt | wc -c

grep -v '^>' B73.fa | tr -d ' \n\r\t' | wc -c


genome_size=2182075994

input_bases=$(awk 'NR % 4 == 2 {sum += length($0)} END {print sum}' /workdir/shared_files/Tx303_HL5WNCCXX_L5_merged_clean.fq)

output_bases=$(zcat /workdir/irk9/data/phg-maize/test2/2x/Tx303.fastq.gz \
  | awk 'NR % 4 == 2 {sum += length($0)} END {print sum}')

echo "Input coverage:"
echo "scale=6; $input_bases / $genome_size" | bc -l

echo "Output coverage:"
echo "scale=6; $output_bases / $genome_size" | bc -l

echo "Observed sampling fraction:"
echo "scale=6; $output_bases / $input_bases" | bc -l


use zcat .fq.gz | wc -l instead of wc -l fq.gz becuase doesn't work on gz files





in_maf="B73_alignment.maf"
out_maf="B73_alignment_swapped.maf"

awk '
function flush_block() {
    if (a_line != "") {
        print a_line

        if (s_count == 2) {
            print s_lines[2]
            print s_lines[1]
        } else {
            for (i = 1; i <= s_count; i++) {
                print s_lines[i]
            }
        }

        for (i = 1; i <= other_count; i++) {
            print other_lines[i]
        }
    }

    a_line = ""
    s_count = 0
    other_count = 0
    delete s_lines
    delete other_lines
}

# Blank line separates MAF blocks
/^$/ {
    flush_block()
    print
    next
}

# Store the a line
$1 == "a" {
    a_line = $0
    next
}

# Store s lines
$1 == "s" {
    s_count++
    s_lines[s_count] = $0
    next
}

# Preserve any other lines in the block
{
    other_count++
    other_lines[other_count] = $0
}

END {
    flush_block()
}
' "$in_maf" > "$out_maf"

echo "Wrote $out_maf"

awk 'BEGIN{OFS="\t"} {print $1, 0, $2}' B73_RefGen_v3.fa.fai > B73_v3.bed








VCF="ref_AF.vcf.gz"

while read -r CHR; do
    echo "Processing ${CHR}"

    OUT_VCF="ref_AF_${CHR}.vcf.gz"

    bcftools index "${OUT_VCF}"

done < chromosomes.txt

echo "Done."



chrs=$(echo chr{1..10} | tr ' ' ',')

bcftools view -t "$chrs" ${DATASET}.vcf.gz \
    -Oz -o ${DATASET}_chrfiltered.vcf.gz

bcftools index -t ${DATASET}_chrfiltered.vcf.gz



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
done



DATASET="ref_AF"

# Run phasing for each chromosome
for CHR in {1..10}; do 

    eagle \
           --vcf ${DATASET}_chr${CHR}_chrfiltered.vcf.gz \
           --chrom chr${CHR} \
           --numThreads=8 \
           --Kpbwt=20000 \
           --outPrefix ${DATASET}_impute_chr${CHR} 
done


VCF="B97.cleaned.vcf.gz"


while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="B97_${CHR}.vcf.gz"


   bcftools view \
       -r "${CHR}" \
       "${VCF}" \
       -Oz \
       -o "${OUT_VCF}"


done < ../ref_panel/chromosomes.txt


echo "Done."


VCF="B97.cleaned.vcf.gz"


while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="B97_${CHR}.vcf.gz"


   bcftools index ${OUT_VCF}


done < ../ref_panel/chromosomes.txt


echo "Done."


TARGET_DIR="/workdir/irk9/data/phg-maize/test_beagle/target"
REF_DIR="/workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr"
OUT_DIR="/workdir/irk9/data/phg-maize/test_beagle/out"

for CHR in {1..10}; do 

    beagle \
        gt=${TARGET_DIR}/B97_chr${CHR}.vcf.gz \
        ref=${REF_DIR}/ref_AF_chr${CHR}.vcf.gz \
        out=${OUT_DIR}/B97_imputed_chr${CHR} \
        nthreads=16 \
        ne=20000 \
        impute=true \
        seed=-99999 
done



bcftools view ../cassava_pangenome_diploid_cleaned.vcf.gz -e 'INFO/AC<3 | INFO/AN-INFO/AC<3' -Ou | \
bcftools norm -m -any -Ou | \
bcftools view -v snps,indels -Ou | \
bcftools norm -f /workdir/smm477/phg-cassava/ref/Mesculenta_671_v8.0.fa -d none -Ou | \
bcftools view -m 2 -M 2 -Ou | \
bcftools view -g ^miss -Oz -o cassava_ref.vcf.gz
bcftools index cassava_ref.vcf.gz
Phase reference
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


  bcftools query -f '%ID\n' cassava_ref.vcf.gz | grep -v '^\.$' | sort | uniq -d | wc -l
AF cleaning
bcftools +fill-tags cassava_ref.vcf.gz -Oz -o cassava_ref_AF.vcf.gz -- -t AF
bcftools index cassava_ref_AF.vcf.gz
echo -e 'CHR\tSNP\tREF\tALT\tAF' > ref_imputation.frq
bcftools query \
  -f '%CHROM\t%CHROM\_%POS\_%REF\_%ALT\t%REF\t%ALT\t%INFO/AF\n' \
  cassava_ref_AF.vcf.gz \
  >> ref_imputation.frq





VCF="cassava_ref_AF.vcf.gz"


while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="/workdir/irk9/data/phg-cassava/truth-vcfs/beagle_ref/ref_split/ref_AF_${CHR}.vcf.gz"


    bcftools index "${OUT_VCF}"

done < chromosomes.txt


echo "Done."
VCF="cassava_ref_AF.vcf.gz"


for i in {01..18}; do
   echo "Processing chr${i}"


   OUT_VCF="ref_AF_chr${i}_renamed.vcf.gz"


   bcftools annotate --rename-chrs chr_rename.txt  -o ${OUT_VCF} ref_AF_Chromosome${i}_chrfiltered.vcf.gz
   bcftools index ${OUT_VCF}

done


echo "Done."


while read -r CHR; do
   INVCF="ref_AF_${CHR}.vcf.gz"
   OUTVCF="ref_AF_${CHR}_chrfiltered.vcf.gz"


   echo "Filtering ${INVCF}"


   bcftools view \
       -t "${CHR}" \
       "${INVCF}" \
       -Oz \
       -o "${OUTVCF}"


   bcftools index -t "${OUTVCF}"
done < chromosomes.txt


echo "Done."











MAP_DIR="/workdir/irk9/data/maps/maize/beagle_split"

for chr in {1..10}; do
    in="${MAP_DIR}/chr${chr}.map"
    out="${MAP_DIR}/chr${chr}.avgdup.map"
    dup_report="${MAP_DIR}/chr${chr}.duplicates.txt"

    if [[ ! -f "$in" ]]; then
        echo "WARNING: missing $in, skipping"
        continue
    fi

    echo "Processing $in"

    awk -v dup_report="$dup_report" '
    BEGIN {
        OFS = "\t"
    }

    {
        key = $1 OFS $4

        chr[key] = $1
        snp[key] = $2
        bp[key] = $4

        sum_cm[key] += $3
        count[key] += 1

        # Keep original input order of first occurrence
        if (!(key in seen)) {
            seen[key] = 1
            order[++n] = key
        }
    }

    END {
        for (i = 1; i <= n; i++) {
            key = order[i]
            avg_cm = sum_cm[key] / count[key]

            print chr[key], snp[key], avg_cm, bp[key]

            if (count[key] > 1) {
                print chr[key], bp[key], count[key], avg_cm > dup_report
            }
        }
    }
    ' "$in" > "$out"

    echo "  wrote deduplicated map: $out"
    echo "  wrote duplicate report: $dup_report"
done




/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/temp_out --reference-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/vcf/out3.vcf 


/workdir/shared_files/maize_test2_answer_key/biokotlin-tools/bin/biokotlin-tools maf-to-gvcf-converter --reference-file=/workdir/smm477/phg-cassava/ref/Mesculenta_671_v8.0.fa --maf-file=/workdir/shared_files/cassava_test2_answer_key/hap2/alignment.maf --output-file=/workdir/shared_files/cassava_test2_answer_key/hap2/hap2.vcf --sample-name=BGM_2098.hap2



bcftools view cassava_test2_merge.vcf.gz | awk '
  BEGIN { OFS="\t" }
  /^##/ { print; next }
  /^#CHROM/ {
      diploid = $10
      sub(/\.hap[12]$/, "", diploid) 
      print "IITA_TMS_IBA30572" "\t" diploid
      next
  }
  {
      new_col = $10 "/" $11

      print $0 "\t" new_col
  }' | bgzip > cassava_diploid.vcf.gz


module load java/21
/workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/phg/bin/phg convert-ropebwt2ps4g-file --ropebwt-bed /workdir/irk9/data/phg-maize/test2/0.01x/grits/matches.bed --spline-knot-dir /workdir/smm477/phg-maize/splines --output-dir /workdir/irk9/data/phg-maize/test2/0.01x/grits/ --min-mem-length 148 --max-num-hits 12


/workdir/irk9/software/ropebwt3/ropebwt3 mem -t4 -l 148 -p 12 /workdir/smm477/phg-maize/ropebwt-index-ML/ropebwt_index.fmd /workdir/irk9/data/phg-maize/test2/0.01x/Tx303.fastq.gz > /workdir/irk9/data/phg-maize/test2/0.01x/grits/matches.bed


/workdir/irk9/software/ropebwt3/ropebwt3 mem -t4 -l 148 -p 12 /workdir/smm477/phg-cassava/ropebwt-index-ML/ropebwt_index.fmd /workdir/irk9/data/phg-cassava/reads/0.01x/VEN25.fastq.gz > /workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/VEN25_matches.bed

/workdir/irk9/software/ropebwt3/ropebwt3 mem -t4 -l 148 -p 12 /workdir/smm477/phg-cassava/ropebwt-index-ML/ropebwt_index.fmd /workdir/irk9/data/phg-cassava/test2/0.01x/Mesc_BGM_2098.fastq.gz > /workdir/irk9/data/phg-cassava/test2/0.01x/grits/matches.bed



/workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/phg_v2/build/distributions/phg/bin/phg convert-ropebwt2ps4g-file --ropebwt-bed /workdir/irk9/data/phg-cassava/target_vcf/2x/grits/VEN25_matches.bed --spline-knot-dir /workdir/smm477/phg-cassava/splines --output-dir /workdir/irk9/data/phg-cassava/target_vcf/2x/grits --min-mem-length 148 --max-num-hits 12

/workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/phg_v2/build/distributions/phg/bin/phg convert-ropebwt2ps4g-file --ropebwt-bed /workdir/irk9/data/phg-cassava/test2/2x/grits/matches.bed --spline-knot-dir /workdir/smm477/phg-cassava/splines --output-dir /workdir/irk9/data/phg-cassava/test2/2x/grits --min-mem-length 148 --max-num-hits 12

/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-maize/test2/2x/grits/out --reference-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-maize/test2/2x/grits/out/out.vcf

/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-cassava/test2/2x/grits/out --reference-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.vcf.gz --out-file /workdir/irk9/data/phg-cassava/test2/2x/grits/out/out.vcf


/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-cassava/target_vcf/2x/grits/out --reference-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.vcf.gz --out-file /workdir/irk9/data/phg-cassava/target_vcf/2x/grits/out/out.vcf


/workdir/irk9/software/ropebwt3/ropebwt3 mem -t4 -l 148 -p 12 /workdir/smm477/phg-cassava/ropebwt-index-ML/ropebwt_index.fmd /workdir/irk9/data/phg-cassava/test2/5.07x/Mesc_BGM_2098.fastq.gz > /workdir/irk9/data/phg-cassava/test2/5.07x/grits/matches.bed

/workdir/irk9/software/ropebwt3/ropebwt3 mem -t4 -l 148 -p 12 /workdir/smm477/phg-maize/ropebwt-index-ML/ropebwt_index.fmd /workdir/irk9/data/phg-maize/test2/5.07x/Tx303.fastq.gz > /workdir/irk9/data/phg-maize/test2/5.07x/grits/matches.bed

/workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/phg_v2/build/distributions/phg/bin/phg convert-ropebwt2ps4g-file --ropebwt-bed /workdir/irk9/data/phg-maize/test2/5.07x/grits/matches.bed --spline-knot-dir /workdir/smm477/phg-maize/splines --output-dir /workdir/irk9/data/phg-maize/test2/5.07x/grits --min-mem-length 148 --max-num-hits 12

/workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/phg_v2/build/distributions/phg/bin/phg convert-ropebwt2ps4g-file --ropebwt-bed /workdir/irk9/data/phg-cassava/test2/5.07x/grits/matches.bed --spline-knot-dir /workdir/smm477/phg-cassava/splines --output-dir /workdir/irk9/data/phg-cassava/test2/5.07x/grits --min-mem-length 148 --max-num-hits 12

/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-maize/test2/5.07x/grits/out --reference-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-maize/test2/5.07x/grits/out/out.vcf



python accuracy.py --truth /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --imputed /workdir/irk9/data/phg-maize/test/impute/imputed/B97_imputed_renamed.vcf -s B97


python /workdir/irk9/data/phg-maize/bellas_scripts/fix_bed_boundaries.py /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt /workdir/irk9/data/phg-maize/target_vcf/1x/grits/vcf/out.bed > /workdir/irk9/data/phg-maize/target_vcf/1x/grits/out/out_updated.bed
module load java/21
/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/out2 --reference-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/out2/out.vcf


Chr end maize

>chr1 308452471
>chr2 243675191
>chr3 238017767
>chr4 250330460
>chr5 226353449
>chr6 181357234
>chr7 185808916
>chr8 182411202
>chr9 163004744
>chr10 152435371



Chr 1 split

0 - 77113117
77113117 - 154226234
154226234 - 231339351
231339351 - 308452471


Chr end cassava

>Chromosome01 42998274
>Chromosome02 38567855
>Chromosome03 33309257
>Chromosome04 35938211
>Chromosome05 33681570
>Chromosome06 30819343
>Chromosome07 34668054
>Chromosome08 41806886
>Chromosome09 37838006
>Chromosome10 32070051
>Chromosome11 33098381
>Chromosome12 37136911
>Chromosome13 36952686
>Chromosome14 29308386
>Chromosome15 32923186
>Chromosome16 34171075
>Chromosome17 35156893
>Chromosome18 33477107



python /workdir/irk9/data/phg-maize/bellas_scripts/fix_bed_boundaries.py /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt /workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/out/VEN25_out.bed > /workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/out2/out.bed

/workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/out2 --reference-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-cassava/target_vcf/0.01x/grits/out2/out.vcf


{
    for i in $(seq 1 18); do
        printf '##contig=<ID=Chromosome%02d>\n' "$i"
    done
    echo '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">'
    echo '##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">'

    
} >



bcftools view maize_pangenome_snps_cleaned_phased_AN_AC_nomissing.vcf.gz -e 'INFO/AC<3 | INFO/AN-INFO/AC<3' -Ou | \
bcftools norm -m -any -Ou | \
bcftools view -v snps -Ou | \
bcftools norm -f /workdir/irk9/data/phg-maize/trial/B73.fa -d none -Ou | \
bcftools view -m 2 -M 2 -Ou | \
bcftools view -g ^miss -Oz -o maize_pangenome_snps_cleaned_beagle.vcf.gz
bcftools index maize_pangenome_snps_cleaned_beagle.vcf.gz

bcftools query -f '%ID\n' maize_pangenome_snps_cleaned_beagle.vcf.gz | grep -v '^\.$' | sort | uniq -d | wc -l

bcftools +fill-tags maize_pangenome_snps_cleaned_beagle.vcf.gz -Oz -o maize_pangenome_snps_cleaned_beagle_AF.vcf.gz -- -t AF
bcftools index maize_pangenome_snps_cleaned_beagle_AF.vcf.gz
echo -e 'CHR\tSNP\tREF\tALT\tAF' > ref_imputation.frq
bcftools query \
  -f '%CHROM\t%CHROM\_%POS\_%REF\_%ALT\t%REF\t%ALT\t%INFO/AF\n' \
  maize_pangenome_snps_cleaned_beagle_AF.vcf.gz \
  >> ref_imputation.frq



VCF="maize_pangenome_snps_cleaned_beagle_AF.vcf.gz"

while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="/workdir/irk9/data/phg-maize/test3/ref_panel/beagle_ref/ref_AF_${CHR}.vcf.gz"


   bcftools view \
       -r "${CHR}" \
       "${VCF}" \
       -Oz \
       -o "${OUT_VCF}"

    bcftools index "${OUT_VCF}"


done < chromosomes.txt


echo "Done."


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
done






#############################


bcftools view cassava_sim_merged_biallelic_diploid_header_AN_AC_nomissing.vcf.gz -e 'INFO/AC<3 | INFO/AN-INFO/AC<3' -Ou | \
bcftools norm -m -any -Ou | \
bcftools view -v snps -Ou | \
bcftools norm -f /workdir/smm477/phg-cassava/ref/Mesculenta_671_v8.0.fa -d none -Ou | \
bcftools view -m 2 -M 2 -Ou | \
bcftools view -g ^miss -Oz -o cassava_pangenome_snps_cleaned_beagle.vcf.gz
bcftools index cassava_pangenome_snps_cleaned_beagle.vcf.gz

bcftools query -f '%ID\n' cassava_pangenome_snps_cleaned_beagle.vcf.gz | grep -v '^\.$' | sort | uniq -d | wc -l

bcftools +fill-tags cassava_pangenome_snps_cleaned_beagle.vcf.gz -Oz -o cassava_pangenome_snps_cleaned_beagle_AF.vcf.gz -- -t AF
bcftools index cassava_pangenome_snps_cleaned_beagle_AF.vcf.gz
echo -e 'CHR\tSNP\tREF\tALT\tAF' > ref_imputation.frq
bcftools query \
  -f '%CHROM\t%CHROM\_%POS\_%REF\_%ALT\t%REF\t%ALT\t%INFO/AF\n' \
  cassava_pangenome_snps_cleaned_beagle_AF.vcf.gz \
  >> ref_imputation.frq





VCF="cassava_pangenome_snps_cleaned_beagle_AF_rename_chr.vcf.gz"

while read -r CHR; do
   echo "Processing ${CHR}"


   OUT_VCF="/workdir/irk9/data/phg-cassava/test3/ref_panel/beagle_ref/ref_AF_${CHR}.vcf.gz"


   bcftools view \
       -r "${CHR}" \
       "${VCF}" \
       -Oz \
       -o "${OUT_VCF}"

    bcftools index "${OUT_VCF}"


done < chromosomes.txt


echo "Done."


for CHR in chr{1..18}; do
   INVCF="ref_AF_${CHR}.vcf.gz"
   OUTVCF="ref_AF_${CHR}_chrfiltered.vcf.gz"

   echo "Filtering ${INVCF}"

   bcftools view \
       -t "${CHR}" \
       "${INVCF}" \
       -Oz \
       -o "${OUTVCF}"

   bcftools index -t "${OUTVCF}"
done





