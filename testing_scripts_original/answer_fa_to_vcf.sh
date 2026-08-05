############
# DONT USE #
############


#!/bin/bash
set -uo pipefail

# -----------------------------
# User-provided paths
# -----------------------------

ANSWER_FA_DIR="/workdir/irk9/data/phg-maize/fastas"
REF_FASTA="/workdir/irk9/data/phg-maize/trial/B73.fa"
TRUTH_VCF_DIR="/workdir/irk9/data/phg-maize/answer_vcf"
THREADS=8

LOG_DIR="$TRUTH_VCF_DIR/logs"

mkdir -p "$TRUTH_VCF_DIR" "$LOG_DIR"

for answer_fasta in "$ANSWER_FA_DIR"/*.fa
do
    [ -e "$answer_fasta" ] || continue

    sample=$(get_base_name "$answer_fasta")

    log_file="$TRUTH_VCF_DIR/${sample}.truth_vcf.log"
    truth_bam="$TRUTH_VCF_DIR/${sample}.answer_aligned.sorted.bam"
    truth_vcf="$TRUTH_VCF_DIR/${sample}.truth.vcf.gz"

    if minimap2 -t "$THREADS" -ax asm5 "$REF_FASTA" "$answer_fasta" 2>> "$log_file" \
        | samtools sort -@ "$THREADS" -o "$truth_bam" - 2>> "$log_file"
    then
        echo "Finished minimap2 | samtools sort"
    else
        echo "ERROR: minimap2 | samtools sort failed for $sample"
        continue
    fi

    if [ ! -s "$truth_bam" ]; then
        echo "ERROR: truth BAM is empty: $truth_bam"
        continue
    fi

    if samtools index "$truth_bam" 2>> "$log_file"; then
        echo "Finished samtools index"
    else
        echo "ERROR: samtools index failed for $sample"
        continue
    fi

    if bcftools mpileup -f "$REF_FASTA" "$truth_bam" -Ou 2>> "$log_file" \
        | bcftools call -mv -Oz -o "$truth_vcf" 2>> "$log_file"
    then
        echo "Finished bcftools mpileup/call"
    else
        echo "ERROR: bcftools mpileup/call failed for $sample"
        continue
    fi

    if bcftools index "$truth_vcf" 2>> "$log_file"; then
        echo "Finished bcftools index"
    else
        echo "ERROR: bcftools index failed for $sample"
        continue
    fi

    echo "Finished truth VCF for sample: $sample"
done