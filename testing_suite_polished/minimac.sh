#!/usr/bin/env bash

set -uo pipefail

############################################
# Minimac4 imputation pipeline
# Assumes:
#   - target files are *.vcf.gz (cleaned VCFs from clean.sh)
#   - reference file is a pre-built .msav file
#   - minimac4 and bcftools are available in PATH
############################################

usage() {
    cat <<EOF
Usage:
  $0 --target-dir TARGET_DIR --out-dir OUT_DIR --ref-msav REF_MSAV

Required arguments:
  --target-dir   Directory containing *.vcf.gz target files (output of clean.sh)
  --out-dir      Directory where output files will be written
  --ref-msav     Pre-built reference MSAV file, e.g. ref.msav

Optional:
  --map FILE     Genetic map file; when provided, passes -m to minimac4
  --threads INT  Number of threads (default: 5)
  --help         Show this help message

Example:
  $0 \\
    --target-dir /path/to/cleaned_vcfs \\
    --out-dir /path/to/imputed_outputs \\
    --ref-msav /path/to/ref.msav
EOF
}

export PATH=/workdir/irk9/software/minimac4/bin:$PATH


TARGET_DIR=""
OUT_DIR=""
REF_MSAV=""
LOG_FILE=""
THREADS=5
MAP=""

# ./minimac.sh --target-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target --out-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/minimac --ref_msav /workdir/irk9/data/phg-maize/truth-vcfs/maize_pangenome_snps.msav 

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --ref-msav)
            REF_MSAV="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --map)
            MAP="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_DIR" || -z "$OUT_DIR" || -z "$REF_MSAV" ]]; then
    echo "ERROR: Missing required arguments."
    usage
    exit 1
fi

mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/log"

mkdir -p "$LOG_FILE"

LOG_FILE="$LOG_FILE/minimac4_pipeline.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_cmd() {
    log "Running: $*"
    "$@" >> "$LOG_FILE" 2>&1
}

log "============================================"
log "Starting minimac4 pipeline"
log "============================================"
log "Input arguments:"
log "TARGET_DIR = $TARGET_DIR"
log "OUT_DIR    = $OUT_DIR"
log "REF_MSAV   = $REF_MSAV"
log "LOG_FILE   = $LOG_FILE"
log "============================================"

if [[ ! -d "$TARGET_DIR" ]]; then
    log "ERROR: Target directory does not exist: $TARGET_DIR"
    exit 1
fi

if [[ ! -f "$REF_MSAV" ]]; then
    log "ERROR: Reference MSAV does not exist: $REF_MSAV"
    exit 1
fi

if ! command -v minimac4 >/dev/null 2>&1; then
    log "ERROR: minimac4 not found in PATH"
    exit 1
fi

if ! command -v bcftools >/dev/null 2>&1; then
    log "ERROR: bcftools not found in PATH"
    exit 1
fi


############################################
# Step 2: Process all target VCF files
############################################

shopt -s nullglob
TARGET_FILES=("$TARGET_DIR"/*.vcf.gz)

if [[ ${#TARGET_FILES[@]} -eq 0 ]]; then
    log "ERROR: No .vcf.gz target files found in $TARGET_DIR"
    exit 1
fi

log "Found ${#TARGET_FILES[@]} target VCF file(s)."

for TARGET_VCF in "${TARGET_FILES[@]}"; do
    TARGET_NAME=$(basename "$TARGET_VCF")
    TARGET_BASE=${TARGET_NAME%.vcf.gz}

    TARGET_BCF="$OUT_DIR/${TARGET_BASE}.bcf"

    log "--------------------------------------------"
    log "Processing target file: $TARGET_VCF"
    log "Target base name: $TARGET_BASE"

    ############################################
    # Step 2a: Convert target VCF to BCF
    ############################################

    if [[ -f "$TARGET_BCF" ]]; then
        log "Target BCF already exists. Skipping VCF-to-BCF conversion: $TARGET_BCF"
    else
        log "Converting target VCF to BCF"
        run_cmd bcftools view --threads "$THREADS" -Ob -o "$TARGET_BCF" "$TARGET_VCF"
    fi

    ############################################
    # Step 2b: Index target BCF
    ############################################

    if [[ -f "${TARGET_BCF}.csi" || -f "${TARGET_BCF}.bai" ]]; then
        log "Target BCF index already exists. Skipping indexing: $TARGET_BCF"
    else
        log "Indexing target BCF"
        run_cmd bcftools index --threads "$THREADS" "$TARGET_BCF"
    fi

    ############################################
    # Step 2c: Find chromosomes in target BCF
    ############################################

    CHR_LIST="$OUT_DIR/${TARGET_BASE}_chromosomes.txt"

    log "Finding chromosomes in target BCF"
    run_cmd bcftools query -f '%CHROM\n' "$TARGET_BCF"

    bcftools query -f '%CHROM\n' "$TARGET_BCF" \
        | sort -u \
        > "$CHR_LIST" 2>> "$LOG_FILE"

    log "Chromosome list written to: $CHR_LIST"
    log "Chromosomes found: $(tr '\n' ' ' < "$CHR_LIST")"

    ############################################
    # Step 2d: Run minimac4 for each chromosome
    ############################################

    while read -r CHR; do
        if [[ -z "$CHR" ]]; then
            continue
        fi

        SAFE_CHR=$(echo "$CHR" | sed 's/[^A-Za-z0-9_.-]/_/g')

        OUT_VCF="$OUT_DIR/${TARGET_BASE}_imputed_${SAFE_CHR}.vcf.gz"

        log "Running minimac4 for target $TARGET_BASE chromosome $CHR"
        log "Output VCF: $OUT_VCF"
        
        if [[ -n "$MAP" ]]; then
            if [[ ! -f "$MAP" ]]; then
                log "ERROR: Map file does not exist: $MAP"
                exit 1
            fi
            log "Using map file: $MAP"
            run_cmd minimac4 \
                "$REF_MSAV" \
                "$TARGET_BCF" \
                --threads "$THREADS" \
                -f GT \
                -m "$MAP" \
                -o "$OUT_VCF" \
                --region "$CHR"
        else
            log "No map file provided; running without map"
            run_cmd minimac4 \
                "$REF_MSAV" \
                "$TARGET_BCF" \
                --threads "$THREADS" \
                -f GT \
                -o "$OUT_VCF" \
                --region "$CHR"
        fi

    done < "$CHR_LIST"

    log "Finished processing target file: $TARGET_VCF"
done

log "============================================"
log "Minimac4 pipeline completed successfully"
log "============================================"