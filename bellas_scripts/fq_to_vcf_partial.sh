#!/bin/bash
set -uo pipefail

# -----------------------------
# User-provided paths
# -----------------------------


#EXAMPLE INPUTS
#READS_DIR="/workdir/irk9/data/phg-maize/reads/1x"
#REF_MAYBE_DIR="/workdir/irk9/data/phg-maize/trial/B73.fa"
#OUT_DIR="/workdir/irk9/data/phg-maize/target_vcf/1x"
#THREADS=2
#Mo18W,Ms71,NC350,NC358,Oh43,Oh7B,P39,Tzi8


READS_DIR=""
REF_MAYBE_DIR=""
OUT_DIR=""
LIST=""
THREADS=2

while [[ $# -gt 0 ]]
do
    case "$1" in
        --reads)
            READS_DIR="$2"
            shift 2
            ;;

        --coordinate_ref)
            REF_MAYBE_DIR="$2"
            shift 2
            ;;

        --out)
            OUT_DIR="$2"
            shift 2
            ;;

        --threads)
            THREADS="$2"
            shift 2
            ;;
        --list)
            IFS=',' read -r -a LIST <<< "$2"
            shift 2
            ;;    
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"


# -----------------------------
# Function to get base file name
# -----------------------------

get_base_name() {
    filename=$(basename "$1")

    filename=${filename%.gz}
    filename=${filename%.fa}
    filename=${filename%.fasta}
    filename=${filename%.fq}
    filename=${filename%.fastq}

    echo "$filename"
}

log_msg() {
    log_file="$1"
    msg="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$log_file"
}

# -----------------------------
# Basic checks
# -----------------------------

if [ ! -d "$READS_DIR" ]; then
    echo "ERROR: reads directory does not exist: $READS_DIR"
    exit 1
fi

if [ ! -f "$REF_MAYBE_DIR" ]; then
    echo "ERROR: reference FASTA does not exist: $REF_MAYBE_DIR"
    exit 1
fi

if [ ! -f "${REF_MAYBE_DIR}.fai" ]; then
    echo "ERROR: FASTA index missing: ${REF_MAYBE_DIR}.fai"
    echo "Run: samtools faidx $REF_MAYBE_DIR"
    exit 1
fi

if [ ! -f "${REF_MAYBE_DIR}.bwt" ]; then
    echo "ERROR: BWA index missing for: $REF_MAYBE_DIR"
    echo "Run: bwa index $REF_MAYBE_DIR"
    exit 1
fi

# -----------------------------
# Loop through FASTQ files
# -----------------------------


for reads in "${LIST[@]}"
do
    file="$READS_DIR/$reads.fastq.gz"
    [ -e "$file" ] || continue

    sample="$reads"
    log_file="$LOG_DIR/${sample}.pipeline.log"
    : > "$log_file"

    log_msg "$log_file" "Processing sample: $sample"
    log_msg "$log_file" "Using reference: $REF_MAYBE_DIR"
    log_msg "$log_file" "Using reads: $file"

    sorted_bam="$OUT_DIR/${sample}_sorted.bam"
    target_vcf="$OUT_DIR/${sample}_target.vcf.gz"

    log_msg "$log_file" "Sorted BAM output: $sorted_bam"
    log_msg "$log_file" "Target VCF output: $target_vcf"

    log_msg "$log_file" "Running bwa mem and samtools sort"

    if bwa mem -t "$THREADS" "$REF_MAYBE_DIR" "$file" 2>> "$log_file" \
        | samtools sort -o "$sorted_bam" - 2>> "$log_file"
    then
        log_msg "$log_file" "Finished bwa mem | samtools sort"
    else
        log_msg "$log_file" "ERROR: bwa mem | samtools sort failed for $sample"
        continue
    fi

    if [ ! -s "$sorted_bam" ]; then
        log_msg "$log_file" "ERROR: sorted BAM file is empty: $sorted_bam"
        continue
    fi

    log_msg "$log_file" "Running samtools index"

    if samtools index --threads "$THREADS" "$sorted_bam" 2>> "$log_file"; then
        log_msg "$log_file" "Finished samtools index"
    else
        log_msg "$log_file" "ERROR: samtools index failed for $sample"
        continue
    fi

    # -----------------------------
    # Variant calling
    # -----------------------------

    log_msg "$log_file" "Running bcftools mpileup and bcftools call"

    if bcftools mpileup --threads "$THREADS" -f "$REF_MAYBE_DIR" "$sorted_bam" -Ou 2>> "$log_file" \
        | bcftools call --threads "$THREADS" -mv -Oz -o "$target_vcf" 2>> "$log_file"
    then
        log_msg "$log_file" "Finished bcftools mpileup/call"
    else
        log_msg "$log_file" "ERROR: bcftools mpileup/call failed for $sample"
        continue
    fi

    if [ ! -s "$target_vcf" ]; then
        log_msg "$log_file" "ERROR: target VCF is empty: $target_vcf"
        continue
    fi

    log_msg "$log_file" "Running bcftools index"

    if bcftools index --threads "$THREADS" "$target_vcf" 2>> "$log_file"; then
        log_msg "$log_file" "Finished bcftools index"
    else
        log_msg "$log_file" "ERROR: bcftools index failed for $sample"
        continue
    fi

    log_msg "$log_file" "Finished sample: $sample"
    log_msg "$log_file" "Output VCF: $target_vcf"

    echo "Finished sample: $sample"
    echo "Output VCF: $target_vcf"
    echo
done

echo "Finished processing all files! :)"