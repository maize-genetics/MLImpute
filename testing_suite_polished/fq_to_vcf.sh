#!/bin/bash
set -uo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --reads DIR --coordinate_ref FASTA --out DIR [OPTIONS]

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
  # Process all samples in a directory:
  $(basename "$0") --reads /data/reads --coordinate_ref /data/ref.fa --out /data/output --threads 15

  # Process specific samples only:
  $(basename "$0") --reads /data/reads --coordinate_ref /data/ref.fa --out /data/output --list Mo18W,Ms71,NC350
EOF
}

# ./fq_to_vcf.sh --reads /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/reads --coordinate_ref  /workdir/irk9/data/phg-maize/trial/B73.fa --out /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target


READS_DIR=""
REF_FILE=""
OUT_DIR=""
THREADS=5
LIST=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reads)          READS_DIR="$2";                    shift 2 ;;
        --coordinate_ref) REF_FILE="$2";                shift 2 ;;
        --out)            OUT_DIR="$2";                      shift 2 ;;
        --threads)        THREADS="$2";                      shift 2 ;;
        --list)           IFS=',' read -r -a LIST <<< "$2"; shift 2 ;;
        --help|-h)        usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# --- Validate required args before touching the filesystem ---

if [[ -z "$READS_DIR" || -z "$REF_FILE" || -z "$OUT_DIR" ]]; then
    echo "ERROR: --reads, --coordinate_ref, and --out are required"
    usage
    exit 1
fi

if [[ ! -d "$READS_DIR" ]]; then
    echo "ERROR: reads directory does not exist: $READS_DIR"
    exit 1
fi

if [[ ! -f "$REF_FILE" ]]; then
    echo "ERROR: reference FASTA does not exist: $REF_FILE"
    exit 1
fi

if [[ ! -f "${REF_FILE}.fai" ]]; then
    echo "ERROR: FASTA index missing: ${REF_FILE}.fai"
    echo "Run: samtools faidx $REF_FILE"
    exit 1
fi

if [[ ! -f "${REF_FILE}.bwt" ]]; then
    echo "ERROR: BWA index missing for: $REF_FILE"
    echo "Run: bwa index $REF_FILE"
    exit 1
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- Helper functions ---

get_base_name() {
    local filename
    filename=$(basename "$1")
    filename=${filename%.gz}
    filename=${filename%.fa}
    filename=${filename%.fasta}
    filename=${filename%.fq}
    filename=${filename%.fastq}
    echo "$filename"
}

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $2" | tee -a "$1"
}

# --- Build list of files to process ---

FILES=()
if [[ ${#LIST[@]} -gt 0 ]]; then
    for name in "${LIST[@]}"; do
        f="$READS_DIR/${name}.fastq.gz"
        if [[ -e "$f" ]]; then
            FILES+=("$f")
        else
            echo "WARNING: sample file not found, skipping: $f"
        fi
    done
else
    for f in "$READS_DIR"/*.fastq.gz; do
        [[ -e "$f" ]] && FILES+=("$f")
    done
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: no .fastq.gz files found to process"
    exit 1
fi

echo "Processing ${#FILES[@]} sample(s)..."

# --- Main loop ---

for reads in "${FILES[@]}"; do
    sample=$(get_base_name "$reads")
    log_file="$LOG_DIR/${sample}.pipeline.log"
    : > "$log_file"

    log_msg "$log_file" "Processing sample: $sample"
    log_msg "$log_file" "Using reference: $REF_FILE"
    log_msg "$log_file" "Using reads: $reads"

    sorted_bam="$OUT_DIR/${sample}_sorted.bam"
    target_vcf="$OUT_DIR/${sample}_target.vcf.gz"

    log_msg "$log_file" "Sorted BAM output: $sorted_bam"
    log_msg "$log_file" "Target VCF output: $target_vcf"

    # Skip entire sample if final output already exists
    if [[ -s "$target_vcf" ]]; then
        log_msg "$log_file" "SKIP: target VCF already exists: $target_vcf"
        echo "Skipping $sample (target VCF already exists)"
        continue
    fi

    # --- BWA alignment ---
    if [[ -s "$sorted_bam" ]]; then
        log_msg "$log_file" "SKIP: sorted BAM already exists: $sorted_bam"
    else
        log_msg "$log_file" "Running bwa mem and samtools sort"
        if bwa mem -t "$THREADS" "$REF_FILE" "$reads" 2>> "$log_file" \
            | samtools sort -o "$sorted_bam" - 2>> "$log_file"; then
            log_msg "$log_file" "Finished bwa mem | samtools sort"
        else
            log_msg "$log_file" "ERROR: bwa mem | samtools sort failed for $sample"
            continue
        fi

        if [[ ! -s "$sorted_bam" ]]; then
            log_msg "$log_file" "ERROR: sorted BAM file is empty: $sorted_bam"
            continue
        fi
    fi

    # --- BAM index ---
    if [[ -f "${sorted_bam}.bai" ]]; then
        log_msg "$log_file" "SKIP: BAM index already exists: ${sorted_bam}.bai"
    else
        log_msg "$log_file" "Running samtools index"
        if samtools index --threads "$THREADS" "$sorted_bam" 2>> "$log_file"; then
            log_msg "$log_file" "Finished samtools index"
        else
            log_msg "$log_file" "ERROR: samtools index failed for $sample"
            continue
        fi
    fi

    # --- Variant calling ---
    log_msg "$log_file" "Running bcftools mpileup and bcftools call"
    if bcftools mpileup --threads "$THREADS" -f "$REF_FILE" "$sorted_bam" -Ou 2>> "$log_file" \
        | bcftools call --threads "$THREADS" -mv -Oz -o "$target_vcf" 2>> "$log_file"; then
        log_msg "$log_file" "Finished bcftools mpileup/call"
    else
        log_msg "$log_file" "ERROR: bcftools mpileup/call failed for $sample"
        continue
    fi

    if [[ ! -s "$target_vcf" ]]; then
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

echo "Finished processing all files!"